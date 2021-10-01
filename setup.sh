apt-get update
apt-get install pv

pip install ipywidgets
echo y | conda install -c conda-forge nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager

pip install -r requirements.txt
wandb login