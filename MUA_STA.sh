#!/bin/bash

# Nom de l'environnement et du script
ENV_NAME="MUA_STA"
SCRIPT_NAME="STA_MU_Exec.py"

echo "--- Démarrage de l'analyse MEA ---"

# Vérifie si conda est disponible, sinon essaie de le localiser
if ! command -v conda &> /dev/null
then
    echo "Conda n'est pas dans le PATH. Tentative de chargement via le profil..."
    # On tente de charger les chemins classiques de conda
    [ -f ~/anaconda3/etc/profile.d/conda.sh ] && source ~/anaconda3/etc/profile.d/conda.sh
    [ -f ~/miniconda3/etc/profile.d/conda.sh ] && source ~/miniconda3/etc/profile.d/conda.sh
fi

# Utilisation de 'conda run' qui est beaucoup plus stable pour les scripts
echo "Activation de l'environnement $ENV_NAME et lancement de $SCRIPT_NAME..."
conda run -n $ENV_NAME --no-capture-output python $SCRIPT_NAME

# Garde le terminal ouvert en cas d'erreur
echo ""
echo "Analyse terminée."
read -p "Appuyez sur [Entrée] pour fermer ce terminal..."
