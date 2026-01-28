import numpy as np
import os
import pickle
import warnings
import tkinter as tk
from tkinter import filedialog
from collections import defaultdict
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import matplotlib
import spikeinterface.full as si
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from concurrent.futures import ProcessPoolExecutor

# Configuration graphique
matplotlib.use('qt5agg')
warnings.filterwarnings('ignore')

def save_obj(obj, name):
    """
    Saves a Python object to a pickle file.

    Args:
        obj (any): The object to be serialized.
        name (str): The filename or path. Adds .pkl if not present.
    """
    name = name if name.endswith('.pkl') else name + '.pkl'
    path = os.path.normpath(name)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    """
    Loads a Python object from a pickle file.

    Args:
        name (str): Path to the .pkl file.

    Returns:
        any: The deserialized Python object.
    """
    name = name if name.endswith('.pkl') else name + '.pkl'
    with open(os.path.normpath(name), 'rb') as f:
        return pickle.load(f)
    
def load_data(input_path, channel_id, nb_channels=256, dtype="uint16", voltage_resolution=0.1042, chunk_size=1000000):
    """
    Loads raw binary data for a specific channel using memory mapping.

    Args:
        input_path (str): Path to the binary file.
        channel_id (int): Index of the channel to extract (e.g., trigger channel).
        nb_channels (int): Total number of channels in the recording.
        dtype (str): Data type of the binary file.
        voltage_resolution (float): Conversion factor for voltage.
        chunk_size (int): Number of samples per block for progress tracking.

    Returns:
        tuple: (data_array, nb_samples) where data_array is the extracted channel.
    """
    m = np.memmap(os.path.normpath(input_path), dtype=dtype, mode='r')
    nb_samples = m.size // nb_channels
    data = np.empty(nb_samples, dtype=float)
    
    for i in tqdm(range(0, nb_samples, chunk_size), desc="Loading Trigger Channel"):
        end = min(i + chunk_size, nb_samples)
        block = m[i*nb_channels : end*nb_channels]
        data[i:end] = block[channel_id::nb_channels].astype(float)
    
    data = (data + np.iinfo('int16').min) / voltage_resolution
    return data, nb_samples

def detect_onsets(data, threshold):
    """
    Vectorized detection of rising edge onsets in a signal.

    Args:
        data (np.array): The signal array (e.g., trigger channel).
        threshold (float): Value to define the rising edge trigger.

    Returns:
        np.array: Indices of the detected onsets.
    """
    test = (data[:-1] < threshold) & (data[1:] >= threshold)
    indices = np.where(test)[0]
    
    # Fine-tuning: shift index to the exact start of the rising slope
    pbar = tqdm(total=5, desc="Getting Triggers", leave=False)
    while True:
        to_shift = (indices > 0) & (data[indices - 1] < data[indices])
        if not np.any(to_shift):
            break
        indices[to_shift] -= 1
        pbar.update(1)
    pbar.close()
    return indices

def run_sanity_check(triggers, sampling_rate, maximal_jitter=0.25e-3):
    """
    Checks trigger regularity to detect missing or extra triggers.

    Args:
        triggers (np.array): Indices of detected triggers.
        sampling_rate (int): Hardware sampling rate in Hz.
        maximal_jitter (float): Maximum allowed timing error in seconds.

    Returns:
        np.array: Indices of problematic triggers (errors).
    """
    if len(triggers) < 2: return np.array([])
    inter_triggers = np.diff(triggers)
    val, counts = np.unique(inter_triggers, return_counts=True)
    mode_val = val[np.argmax(counts)]
    errors = np.where(np.abs(inter_triggers - mode_val) >= maximal_jitter * sampling_rate)[0]
    if errors.size > 0:
        print(f"Erreurs triggers : {len(errors)}")
    else:
        print(f"Triggers OK ({len(triggers)})")
    return triggers[errors].astype('int64')

def image_projection(image, setup_id):
    """
    Applies spatial rotation or flipping based on the optical setup geometry.

    Args:
        image (np.array): 2D array representing a stimulus frame.
        setup_id (int): Hardware ID to determine projection orientation.

    Returns:
        np.array: Transformed image.
    """
    if setup_id == 2:
        return np.flipud(np.rot90(image))
    elif setup_id == 3:
        return np.fliplr(image)
    return image

def checkerboard_from_binary(nb_frames, nb_checks, path, setup_id):
    """
    Extracts checkerboard frames from a binary file using bit-unpacking.

    Args:
        nb_frames (int): Total frames to extract.
        nb_checks (int): Squares per side of the checkerboard.
        path (str): Path to the binary stimulus file.
        setup_id (int): Hardware configuration for projection mapping.

    Returns:
        np.array: 3D array (frames, x, y) of stimulus frames.
    """
    total_bits = nb_frames * nb_checks * nb_checks
    nb_bytes = (total_bits + 7) // 8
    
    with open(path, mode='rb') as f:
        raw_data = np.frombuffer(f.read(nb_bytes), dtype=np.uint8)

    all_bits = np.unpackbits(raw_data, bitorder='little')[:total_bits]
    all_frames = all_bits.reshape((nb_frames, nb_checks, nb_checks))
    checkerboard = np.zeros((nb_frames, nb_checks, nb_checks), dtype='uint8')
    
    for frame in tqdm(range(nb_frames), desc="Traitement des frames"):
        image = all_frames[frame].astype(float)
        checkerboard[frame] = image_projection(image, setup_id).astype('uint8')
        
    return checkerboard

def extract_from_sequence(cell_spikes, triggers, nb_repeats, stim_frequency, nb_frames_by_sequence, sequence_portion=(0.5, 1)):
    """
    Segments spike trains into repeated stimulus sequences.

    Args:
        cell_spikes (np.array): Timestamps of spikes in seconds.
        triggers (np.array): Trigger timestamps in seconds.
        nb_repeats (int): Number of experimental repeats.
        stim_frequency (int): Frequency of the stimulus (Hz).
        nb_frames_by_sequence (int): Number of frames per full sequence.
        sequence_portion (tuple): (Start, End) fraction of the sequence to analyze.

    Returns:
        dict: Spike trains, binned counts per frame, and PSTH.
    """
    f0, f1 = sequence_portion
    nb_frames_portion = int((f1 - f0) * nb_frames_by_sequence)
    
    start_indices = np.arange(nb_repeats) * nb_frames_by_sequence + int(f0 * nb_frames_by_sequence)
    end_indices = np.arange(nb_repeats) * nb_frames_by_sequence + int(f1 * nb_frames_by_sequence)
    
    t_starts = triggers[start_indices]
    t_ends = triggers[end_indices]
    
    spike_trains = []
    spikes_counts = np.zeros((nb_repeats, nb_frames_portion))

    relevant_indices = np.searchsorted(cell_spikes, [t_starts.min(), t_ends.max()])
    relevant_spikes = cell_spikes[relevant_indices[0]:relevant_indices[1]]

    for i in range(nb_repeats):
        ts = t_starts[i]
        te = t_ends[i]
        idx_s, idx_e = np.searchsorted(relevant_spikes, [ts, te])
        spike_seq = relevant_spikes[idx_s:idx_e] - ts
        spike_trains.append(spike_seq)
        counts, _ = np.histogram(spike_seq, bins=nb_frames_portion, range=(0, te - ts))
        spikes_counts[i, :] = counts

    return {
        "spike_trains": spike_trains,
        "counted_spikes": spikes_counts,
        "psth": spikes_counts.sum(axis=0) / nb_repeats * stim_frequency
    }

def compute_3D_sta(data, checkerboard, nb_frames_by_sequence, temporal_dimension):
    """
    Computes the Spike-Triggered Average (STA) in 3D (time, x, y).

    Args:
        data (dict): Result from extract_from_sequence containing binned spikes.
        checkerboard (np.array): 3D array of stimulus frames.
        nb_frames_by_sequence (int): Sequence length in frames.
        temporal_dimension (int): Number of frames before a spike to average.

    Returns:
        np.array: 3D STA normalized between -1 and 1.
    """
    nb_frames_half = int(nb_frames_by_sequence / 2)
    spikes = data["counted_spikes"][:, temporal_dimension:nb_frames_half].flatten()
    total_spikes = np.sum(spikes)
    
    if total_spikes == 0:
        return np.zeros((temporal_dimension, checkerboard.shape[1], checkerboard.shape[2]))

    spike_indices = np.where(spikes > 0)[0]
    weights = spikes[spike_indices]
    sta = np.zeros((temporal_dimension, checkerboard.shape[1], checkerboard.shape[2]))
    
    for i, w in zip(spike_indices, weights):
        seq_idx = i // (nb_frames_half - temporal_dimension)
        frame_in_seq = i % (nb_frames_half - temporal_dimension) + temporal_dimension
        start = seq_idx * nb_frames_half + frame_in_seq - temporal_dimension
        end = start + temporal_dimension
        sta += w * checkerboard[start:end, :, :]

    sta /= total_spikes
    sta -= np.mean(sta)
    max_val = np.max(np.abs(sta))
    if max_val > 0:
        sta /= max_val
    return sta

def get_temporal_spatial_sta(sta_3D):
    """
    Extracts the peak spatial frame and the temporal waveform from a 3D STA.

    Args:
        sta_3D (np.array): The 3D STA (time, x, y).

    Returns:
        tuple: (temporal_vector, spatial_frame, peak_coordinates)
    """
    idx_max = np.argmax(np.abs(sta_3D))
    best_t, best_x, best_y = np.unravel_index(idx_max, sta_3D.shape)
    sta_temporal = sta_3D[:, best_x, best_y]
    sta_spatial = sta_3D[best_t, :, :]
    max_spatial = np.max(np.abs(sta_spatial))
    if max_spatial > 0:
        sta_spatial = sta_spatial / max_spatial
    return sta_temporal, sta_spatial, (best_t, best_x, best_y)

def process_single_electrode(args):
    """
    Parallelizable wrapper to process all analysis for one electrode.

    Args:
        args (tuple): Contains (electrode_id, mapping_info, spikes, triggers, parameters).

    Returns:
        tuple: (electrode_id, result_dict)
    """
    electrode, mapping_info, spike_train, triggers, params = args
    nb_repeats = params['nb_repeats']
    stim_freq = params['stim_freq']
    nb_frames = params['nb_frames']
    temp_dim = params['temp_dim']
    checkerboard = params['checkerboard']

    res_r = extract_from_sequence(spike_train, triggers, nb_repeats, stim_freq, nb_frames, (0.5, 1))
    res_s = extract_from_sequence(spike_train, triggers, nb_repeats, stim_freq, nb_frames, (0, 0.5))
    sta_3d = compute_3D_sta(res_s, checkerboard, nb_frames, temp_dim)
    _, sta_spat, _ = get_temporal_spatial_sta(sta_3d)

    return electrode, {
        'raster_spikes': res_r["spike_trains"],
        'sta_spatial': sta_spat
    }

def plot_stitched_sta(data_source, mapping, grid_size=16, padding=3):
    """
    Stitches individual spatial STAs into a high-performance 16x16 grid image.

    Args:
        data_source (dict/list): Processed analysis results.
        mapping (dict): Electrode ID to (row, col) coordinates.
        grid_size (int): Dimensions of the MEA grid (default 16).
        padding (int): Pixel spacing between electrode tiles.
    """
    data_dict = dict(data_source) if isinstance(data_source, list) else data_source
    if not data_dict:
        print("Error: No processed data found.")
        return

    first_elec_id = next(iter(data_dict))
    h, w = data_dict[first_elec_id]['sta_spatial'].shape
    canvas_h = grid_size * h + (grid_size - 1) * padding
    canvas_w = grid_size * w + (grid_size - 1) * padding
    full_canvas = np.full((canvas_h, canvas_w), np.nan) 

    for electrode, (row, col) in mapping.items():
        if electrode not in data_dict: continue
        sta = data_dict[electrode]['sta_spatial'].copy()
        vmax = np.max(np.abs(sta))
        if vmax > 0: sta /= vmax
        y_start = row * (h + padding)
        x_start = col * (w + padding)
        full_canvas[y_start : y_start + h, x_start : x_start + w] = sta

    plt.figure(figsize=(10, 10))
    current_cmap = plt.cm.get_cmap('bwr').copy()
    current_cmap.set_bad(color='white') 
    plt.imshow(full_canvas, cmap=current_cmap, vmin=-1, vmax=1, interpolation='nearest')
    plt.axis('off')
    plt.title(f"Stitched STA Grid ({grid_size}x{grid_size})", fontsize=16, pad=20)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__" : 
    # --- Paramètres d'acquisition ---
    SAMPLING_RATE = 20000
    TOTAL_CHANNELS = 256
    TRIGGER_CHANNEL = 126
    DATA_TYPE = 'uint16'

    # --- Choix du Setup ---
    SETUP = 3  # 1 pour MEA1, 2 pour MEA2, 3 pour Opto

    if SETUP == 1:
        DMD_POLARITY = 1
        PIXEL_SIZE = 2.3
        TRIGGER_THRESHOLD = 150e+3
    elif SETUP == 2:
        DMD_POLARITY = 1
        PIXEL_SIZE = 3.5
        TRIGGER_THRESHOLD = 150e+3
    elif SETUP == 3:
        DMD_POLARITY = -1
        PIXEL_SIZE = 2.8
        TRIGGER_THRESHOLD = 170e+3

    # Sélection fichier
    root = tk.Tk(); root.withdraw()
    raw_path = filedialog.askopenfilename(title='Select a Checkerboard RAW file...')
    if not raw_path: 
        print('File not Found')
        exit()
    else:
        print(f"Selected File : {raw_path}")
    mapping = load_obj('./electrodes_mapping_MEA_MCS_256.pkl')

    # --- Paramètres Stimulus & Analyse ---
    NB_CHECKS = int(input("Number of checks per side : "))
#     NB_CHECKS = 40

    NB_FRAMES_SEQ = 1200
    TEMPORAL_DIM = 30
    PLOT_RASTER = bool(input("Plot raster ? (takes a bit longer) ") in ['yes', 'Yes', 'Y', 'y', 'YES'])
#     PLOT_RASTER = True
    
    
    # 1. Triggers
    print("Lecture triggers...")
    trig_raw, _ = load_data(raw_path, channel_id=TRIGGER_CHANNEL)
    trig_idx = detect_onsets(trig_raw, TRIGGER_THRESHOLD)
    run_sanity_check(trig_idx, SAMPLING_RATE)
    triggers = trig_idx / SAMPLING_RATE
    nb_repeats = len(triggers) // NB_FRAMES_SEQ

    # --- Compute STIM_FREQ dynamically ---
    # Calculate the mean time between consecutive frames (triggers)
    # frequency = 1 / mean_inter_trigger_interval
    avg_dt = np.mean(np.diff(triggers))
    STIM_FREQ = int(round(1.0 / avg_dt))

    print(f"Detected Stimulus Frequency: {STIM_FREQ} Hz")
    print(f"Number of repeats: {nb_repeats}")

    
    # 2. Traitement Spikes
    print("Extraction des spikes...")
    rec = si.read_binary(raw_path, sampling_frequency=SAMPLING_RATE, num_channels=TOTAL_CHANNELS, dtype=DATA_TYPE)

    # FIX: Convert unsigned to signed BEFORE filtering
    # This is mandatory for uint16 recordings in SpikeInterface those i remain unsure of the extend of this modification
    rec_signed = si.unsigned_to_signed(rec)

    rec_filt = si.common_reference(si.bandpass_filter(rec_signed))

    peaks = detect_peaks(rec_filt, method="by_channel", peak_sign="neg", detect_threshold=6, n_jobs=10, progress_bar=True)
    spike_trains_mua = defaultdict(list)
    for p in peaks: spike_trains_mua[p[1]].append(p[0] / SAMPLING_RATE)

    # 3. Stimulus
    print("Chargement stimulus...")
    stim_path = "./binarysource1000Mbits"
    checkerboard = checkerboard_from_binary(nb_repeats * (NB_FRAMES_SEQ // 2), NB_CHECKS, stim_path, SETUP)

    # 4. Compute RASTER and STA
    # Prepare shared parameters
    params = {
        'nb_repeats': nb_repeats,
        'stim_freq': STIM_FREQ,
        'nb_frames': NB_FRAMES_SEQ,
        'temp_dim': TEMPORAL_DIM,
        'checkerboard': checkerboard # Large arrays can be slow to pass between processes
    }

    # Filter tasks
    tasks = [
        (elec, mapping[elec], np.array(spike_trains_mua[elec]), triggers, params)
        for elec in mapping.keys()
        if elec not in [127, 128, 255, 256] and elec in spike_trains_mua
    ]

    processed_data = {}

    # Run in Parallel
    # Adjust max_workers to the number of physical cores you want to use (e.g., 4, 8, or None for all)
    with ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(process_single_electrode, tasks), 
                            total=len(tasks), 
                            desc="Parallel Analysis"))

    # Convert list of tuples back to dictionary
    processed_data = dict(results)    
    
    if PLOT_RASTER:
        fig_r, axs_r = plt.subplots(16, 16, figsize=(8, 8))

        data_dict = dict(processed_data) if isinstance(processed_data, list) else processed_data

        for electrode, (row, col) in tqdm(mapping.items(), desc="Plotting Rasters"):
            ax_r = axs_r[row, col]

            ax_r.set_xticks([])
            ax_r.set_yticks([])
            for spine in ax_r.spines.values():
                spine.set_visible(False)

            if electrode not in data_dict:
                # On laisse les électrodes vides en gris très clair pour voir la grille
                ax_r.set_facecolor('#f9f9f9') 
                continue
            data = data_dict[electrode]
            ax_r.eventplot(data['raster_spikes'], 
                           linewidths=0.1, 
                           alpha = 0.5,
                           color='black', 
                           rasterized=True) 
    
    

        plt.subplots_adjust(wspace=0.1, hspace=0.1, left=0.01, right=0.99, bottom=0.01, top=0.99)
        plt.show(block = False)
        

    plot_stitched_sta(processed_data, mapping)
    
#    input('Press any key to close...')
