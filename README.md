# MEA Spike Analysis & STA Pipeline

This script provides a high-performance solution for processing 256-channel Micro-Electrode Array (MEA) data. It includes automated trigger detection, MUA spike extraction from .raw files using SpikeInterface, and parallelized Spike-Triggered Average (STA) computation. This is meant to be use during an experiment to test retina's visual responses to checkerboard

---

## 🛠 Installation

To ensure all dependencies (especially the GUI backends and multiprocessing handlers) work correctly, follow these steps to create a dedicated Conda environment:

## 🛠️ Quick Setup

To ensure all GUI backends and multiprocessing handlers work correctly, we recommend using the provided `environment.yml` file.

### 1. Create the Environment Open your Terminal or Anaconda Prompt in the project folder and run:

```bash conda env create -f environment.yml ```

### 2. Activate the Environment 

```bash conda activate mea_analysis ```

### 3. (Optional) Register as Jupyter Kernel If you plan to use this environment with Jupyter Notebooks:

```bash conda install ipykernel -y python -m ipykernel install --user --name mea_analysis --display-name "Python 3.10 (MEA)" ```

### 4. Configuration
Before running the analysis, you must configure the script to match your hardware parameters:

1. Open MUA_STA_Exec.py in your text editor.
2. Locate the Global Variables section at the top of the file.
3. Update the SETUP variable and hardware parameters as needed:

# Example Global Variables to adjust:
# SETUP = "Your_Setup_Name"
# TOTAL_CHANNELS = 64
# TRIGGER_CHANNEL = 1

Note: Once these variables are saved, you will not need to adjust them again for this specific hardware setup.
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

