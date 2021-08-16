# pip install gdown

# download our prepared dataset 
gdown https://drive.google.com/uc?id=1qx-FRJV0yZmiF0GzQLsZFSJznUgNipf7 -O data/
gdown https://drive.google.com/uc?id=1Pz3wEpp6E7aFSUg5zh1Nm6EpWmRG_HnY -O data/

# download the pretrained latent classifier 
gdown https://drive.google.com/uc?id=1K_ShWBfTOCbxBcJfzti7vlYGmRbjXTfn -O models/

# download the pretrained models
gdown https://drive.google.com/uc?id=14uipafI5mena7LFFtvPh6r5HdzjBqFEt -O logs/
unzip logs/pretrained_models.zip -d logs/