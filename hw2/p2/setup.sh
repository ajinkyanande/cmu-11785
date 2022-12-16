# 1) download all files
# 2) sh setup.sh
# 3) python hw2p2.py

# update
sudo apt update
sudo apt -y upgrade

# install requirements
pip3 install -r requirements.txt
source ~/.bashrc

# kaggle account info
mkdir ~/.kaggle
jq -n '{"username": "ajinkyanande", "key": "5f60b9bc169fe67552e51c70e754066d"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# data folder
mkdir data
cd data

# download classification data
kaggle competitions download -c 11-785-f22-hw2p2-classification
unzip -qo '11-785-f22-hw2p2-classification.zip' -d ''
mv 11-785-f22-hw2p2-classification/classification classification

# download verification data
kaggle competitions download -c 11-785-f22-hw2p2-verification
unzip -qo '11-785-f22-hw2p2-verification.zip' -d ''

# file tree :
    # 11785-hw2p2
        # requirements.txt
        # hw2p2.py
        # data
        #     classification
        #     verification
