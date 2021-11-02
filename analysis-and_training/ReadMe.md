# Handwritten Digit Recognition

## Virtual Environement (P1cv_venv)

* Create V_env : conda create -n P1cv_venv
* ativate V_env : conda activate P1cv_venv
* Libraries : requirements.txt
* install libraries : conda install -r requirements.txt
* Chest Installed Libraries : conda list

## Kernel
* pip install ipykernel
* python -m ipykernel install --user --name P1cv_venv --display-name "PP1cv_venv_k"

## Sharing conda environment as a YAML file
* conda env export >> P1cv_venv.yml
* conda env create -f P1cv_venv.yml