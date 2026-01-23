# MEA Spike Analysis & STA Pipeline

This script provides a high-performance solution for processing 256-channel Micro-Electrode Array (MEA) data. It includes automated trigger detection, MUA spike extraction from .raw files using SpikeInterface, and parallelized Spike-Triggered Average (STA) computation. This is meant to be use during an experiment to test retina's visual responses to checkerboard

---

## 🛠 Installation

To ensure all dependencies (especially the GUI backends and multiprocessing handlers) work correctly, follow these steps to create a dedicated Conda environment:

1. **Open your Terminal** (Linux) or **Anaconda Prompt** (Windows).
2. **Create the environment**:
   conda create -n mea_analysis python=3.10 -y
   
3. **Activate the environment**:
   conda activate mea_analysis
   
4. **Install dependencies**:
   conda install numpy matplotlib tqdm -y
   pip install spikeinterface[full] PyQt5

5. **Setup**:
   Open MUA_STA_Exec.py and tune the SETUP global variable to your setup. You may also need to adapt other global variables (TOTAL_CHANNELS,TRIGGER_CHANNEL)
   Once done you won't need it to do it for this setup
---

## 📂 Required Files

Ensure the following files are in the same directory as your script:
* MUA_STA_Exec.py
* electrodes_mapping_MEA_MCS_256.pkl (The electrode geometry file)
* binarysource1000Mbits (The stimulus binary source)

---

## 🚀 How to Run

### 🐧 Linux
1. Make the launch script executable: chmod +x MUA_STA_Exec.sh
2. Run the script: ./MUA_STA_Exec.sh

### 🪟 Windows
1. Double-click MUA_STA_Exec.bat

---

## 📈 Script Workflow

1. Trigger Detection: The script loads the trigger channel and identifies onsets based on a hardware-specific threshold (MEA1, MEA2, or MEA3).
2. Spike Extraction: Uses spikeinterface to filter data and detect peaks (MUA), optimized for 10 CPU cores.
3. Parallel Analysis: Computes Raster data and 3D STA simultaneously across all electrodes using ProcessPoolExecutor with 10 CPU cores.
4. Visualization:
    - Raster Plot: Generates a 16x16 grid showing the spiking activity of the entire array (if specified).
    - Stitched STA: A high-speed visualization that tiles all spatial receptive fields into a single high-resolution image.

---

## ⚠️ Notes
* Memory: High-resolution checkerboards require significant RAM when processed in parallel.
* GUI: The script uses Qt5Agg. If you are running this over SSH, ensure X11 forwarding is enabled.

