#!/bin/bash
# Launch the MUA_STA GUI in its conda environment (creates the env on first use).
# The GUI is started detached, so the terminal that ran this script can close immediately.
# Output of the GUI goes to ~/.mua_sta/mua_sta.log.
parent_path=$(cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P)
cd "$parent_path"

if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi

if ! conda env list | grep -qE "^mua_sta[[:space:]]"; then
    echo "Environment 'mua_sta' not found: creating it from environment.yml (a few minutes)..."
    conda env create -f environment.yml || { echo "Environment creation failed."; read -p "Press Enter to close."; exit 1; }
fi

conda activate mua_sta
mkdir -p "$HOME/.mua_sta"
setsid python -m mua_sta > "$HOME/.mua_sta/mua_sta.log" 2>&1 < /dev/null &
disown
