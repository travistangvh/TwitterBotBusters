# Use conda anaconda3/2023.3
# conda create --name botbusters python==3.7.1 
conda create --name botbusters python==3.9
conda activate botbusters
# conda install --name botbusters pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch -y
# conda install pytorch::pytorch torchvision torchaudio -c pytorch -y
conda install --name botbusters pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
# conda install --name botbusters pytorch-sparse -c pyg
conda install --name botbusters -c huggingface tokenizers -y
conda install --name botbusters -c conda-forge transformers==4.4.2 -y
conda install --name botbusters importlib-metadata -y
conda install --name botbusters packaging==21.3 -y
conda install --name botbusters tqdm -y
conda install --name botbusters -c conda-forge matplotlib -y
conda install --name botbusters scikit-learn -y
# conda install --name botbusters pyg -c pyg -c conda-forge -y
pip install --upgrade --force-reinstall torch -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install --upgrade --force-reinstall torch-geometric -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install --upgrade --force-reinstall torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install --upgrade --force-reinstall torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install pyyaml
pip install pandas==1.3.5