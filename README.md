# Clone Repo
git clone https://github.com/Esperanto-mega/explainer.git
# Train GCN
python model_training.py --data_path ./data --model_path ./ba2motif.pth --device cuda:0
# Train PGExplainer
python explainer_training.py --data_path ./data --model_path ./ba2motif.pth --device cuda:0
