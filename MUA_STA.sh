#!/bin/bash
# Launch the MUA_STA GUI in its conda environment, creating the environment on first use.
#
# The environment's interpreter is called by its full path rather than through `conda activate`
# + `python`: a machine whose profile puts another environment's bin early on PATH (which is
# easy to end up with) would otherwise run the wrong Python, and the only sign of it would be
# an ImportError buried in the log.
#
# The GUI is started detached, so the terminal that ran this can close immediately, and its
# output goes to ~/.mua_sta/mua_sta.log.
#
#   ./MUA_STA.sh                      launch
#   ./MUA_STA.sh --install-shortcut   add it to this machine's application menu
set -u
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "$here"

ENV_NAME=mua_sta

die() {
    echo "$1" >&2
    [ -t 0 ] && read -r -p "Press Enter to close."
    exit 1
}

# --- find conda -----------------------------------------------------------------------
conda_sh=""
if command -v conda >/dev/null 2>&1; then
    conda_sh="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [ ! -f "$conda_sh" ]; then
    for root in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" \
                "$HOME/mambaforge" "/opt/conda"; do
        if [ -f "$root/etc/profile.d/conda.sh" ]; then
            conda_sh="$root/etc/profile.d/conda.sh"
            break
        fi
    done
fi
[ -f "$conda_sh" ] || die "Conda not found. Install Anaconda or Miniconda, then run this again."
# shellcheck disable=SC1090
source "$conda_sh"

# --- create the environment the first time --------------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' not found: creating it from environment.yml (a few minutes)..."
    conda env create -f environment.yml || die "Creating the environment failed."
fi

prefix=$(conda env list | awk -v name="$ENV_NAME" '$1 == name {print $NF}')
python_bin="$prefix/bin/python"
[ -x "$python_bin" ] || die "The '$ENV_NAME' environment has no python at $python_bin."

# --- say so now if the environment is broken, rather than in a log nobody reads --------
if ! "$python_bin" -c "import PyQt6, pyqtgraph, numpy, scipy, spikeinterface" 2>/tmp/mua_sta_check.$$; then
    echo "The '$ENV_NAME' environment is missing something:" >&2
    tail -3 "/tmp/mua_sta_check.$$" >&2
    rm -f "/tmp/mua_sta_check.$$"
    die "Delete it (conda env remove -n $ENV_NAME) and run this again to rebuild it."
fi
rm -f "/tmp/mua_sta_check.$$"

# --- the application-menu shortcut ----------------------------------------------------
if [ "${1:-}" = "--install-shortcut" ]; then
    target="$HOME/.local/share/applications/mua_sta.desktop"
    mkdir -p "$(dirname "$target")"
    sed -e "s|^Exec=.*|Exec=bash \"$here/MUA_STA.sh\"|" \
        -e "s|^Path=.*|Path=$here|" "$here/MUA_STA.desktop" > "$target"
    chmod +x "$target"
    update-desktop-database "$(dirname "$target")" 2>/dev/null
    echo "Shortcut installed: $target"
    exit 0
fi

# --- launch ---------------------------------------------------------------------------
mkdir -p "$HOME/.mua_sta"
setsid "$python_bin" -m mua_sta > "$HOME/.mua_sta/mua_sta.log" 2>&1 < /dev/null &
disown
echo "MUA_STA started (log: $HOME/.mua_sta/mua_sta.log)"
