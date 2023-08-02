# TwitterBotBusters

### Introduction

Bots are a prevalent problem on Twitter. At best, bots create inauthentic interactions and artificially inflate one’s social influence; at worst, they spread dangerous content like scams or fake news. At Twitter’s scale, it can no longer rely on  human annotators to identify bots from humans and has to opt for some form of automatic detection. This project aims to detect human from bots using their user description and tweets modeled with different deep learning approaches, including multilayer perceptron (MLP) and different types of graph neural networks (GNN), including graph convolutional network (GCN), graph isomorphic network (GIN), and graph attention network (GAN). We also experimented with different model architectures for extracting the embedding that summarizes the users' tweets. We found that the best model is an architecture that combines MLP and GAN, giving an accuracy score of X on the Y dataset.

### Dataset Format

Cresci-15 dataset contains `node.json`, `label.csv`, `split.csv` and `edge.csv` (for datasets with graph structure). 

### How to download Cresci-15 dataset

Cresci-15 is available at [Google Drive](https://drive.google.com/drive/folders/1gXFZp3m7TTU-wyZRUiLHdf_sIZpISrze). 

1. Download `Other-Dataset-TwiBot22-Format.zip` and unzip.
2. Copy `cresci-2015` to `src/BotRGCN/datasets/`.

### Requirements

To setup the environment and install the requirement `bash commands_local.sh`. You might need to adjust the cuda version depending on the cuda version that you use. 

### How to run baselines

1. clone this repo by running `git clone https://github.com/travistangvh/TwitterBotBusters`
2. change directory to `src/BotRGCN/datasets` and download datasets and create new folder in `./cresci-2015`
3. create the preprocessed data by changing the directory to `src/BotRGCN/cresci_15` and run `python3 ./preprocess_combined.py`. This will create a preprocess data in the `src/BotRGCN/cresci_15/processed` 
3. change directory to `src/GCN_GAT`
4. run experiments by executing `python train.py --config gat-mlp-1.yaml`. You can explore other model by changing the config file. 