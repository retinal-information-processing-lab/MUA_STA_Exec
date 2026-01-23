# MEA Spike Analysis & STA Pipeline

This script provides a high-performance solution for processing 256-channel Micro-Electrode Array (MEA) data. It includes automated trigger detection, MUA spike extraction from .raw files using SpikeInterface, and parallelized Spike-Triggered Average (STA) computation. This is meant to be used during an experiment to test retina's visual responses to checkerboard stimuli.

---

## 🛠 Installation

To ensure all GUI backends and multiprocessing handlers work correctly, we recommend using the provided `environment.yml` file.

### 1. Create the Environment
Open your **Terminal** or **Anaconda Prompt** in the project folder and run:

conda env create -f environment.yml

### 2. Activate the Environment
conda activate mea_analysis

### 3. (Optional) Jupyter Kernel Setup
If you plan to use this environment with Jupyter Notebooks:

conda install ipykernel -y
python -m ipykernel install --user --name mea_analysis --display-name "Python 3.10 (MEA)"

### 4. Configuration
Before running the analysis, you must configure the script to match your hardware parameters:

1. Open `MUA_STA_Exec.py` in your text editor.
2. Locate the **Global Variables** section at the top of the file.
3. Update the `SETUP`, `TOTAL_CHANNELS`, and `TRIGGER_CHANNEL` variables as needed.

**Example Configuration:**

# MUA_STA_Exec.py (Global Variables Section)
# ----------------------------------------
# SETUP = "MEA2"           # Options: MEA1, MEA2, MEA3
# TOTAL_CHANNELS = 256     # Total number of channels in your recording
# TRIGGER_CHANNEL = 256    # The specific channel used for the stimulus trigger

Note: Once these variables are saved, you will not need to modify them again unless your hardware setup changes.

---

## 📂 Required Files

Ensure the following files are in the same directory as your script:
* **MUA_STA_Exec.py**: The main execution script.
* **electrodes_mapping_MEA_MCS_256.pkl**: The electrode geometry file.
* **binarysource1000Mbits**: The stimulus binary source file.

---

## 🚀 How to Run

### 🐧 Linux
1. Make the launch script executable: `chmod +x MUA_STA_Exec.sh`
2. Run the script: `./MUA_STA_Exec.sh`

### 🪟 Windows
1. Double-click `MUA_STA_Exec.bat`

---

## 📈 Script Workflow

1. **Trigger Detection**: Identifies onsets based on hardware-specific thresholds (MEA1, MEA2, or MEA3).
2. **Spike Extraction**: Uses `spikeinterface` to filter data and detect MUA peaks, optimized for 10 CPU cores.
3. **Parallel Analysis**: Computes Raster data and 3D STA simultaneously using `ProcessPoolExecutor` with 10 CPU cores.
4. **Visualization**:
    * **Raster Plot**: A 16x16 grid showing spiking activity across the entire array.
    * **Stitched STA**: A high-speed visualization tiling all spatial receptive fields into a single high-resolution image.

---

## ⚠️ Notes
* **Memory**: High-resolution checkerboards require significant RAM when processed in parallel.
* **GUI**: The script uses `Qt5Agg`. If running over SSH, ensure X11 forwarding is enabled.
