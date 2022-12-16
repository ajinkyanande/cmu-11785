# 1) download all files
# 2) sh setup.sh
# 3) python hw4p2.py

# update
sudo apt update
sudo apt upgrade

# install requirements
sudo apt install python3-pip
pip3 install -r requirements.txt
source ~/.bashrc

# kaggle account info
mkdir ~/.kaggle
jq -n '{"username": "ajinkyanande", "key": "5f60b9bc169fe67552e51c70e754066d"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# data folder
mkdir data
cd data

# toy data
mkdir toy
cd toy
wget -q https://cmu.box.com/shared/static/wok08c2z2dp4clufhy79c5ee6jx3pyj9 --content-disposition --show-progress
wget -q https://cmu.box.com/shared/static/zctr6mvh7npfn01forli8n45duhp2g85 --content-disposition --show-progress
wget -q https://cmu.box.com/shared/static/m2oaek69145ljeu6srtbbb7k0ip6yfup --content-disposition --show-progress
wget -q https://cmu.box.com/shared/static/owrjy0tqra3v7zq2ru7mocy2djskydy9 --content-disposition --show-progress
cd ..

# full data
mkdir full
cd full
kaggle competitions download -c 11-785-f22-hw4p2
unzip -qo '11-785-f22-hw4p2.zip' -d ''
cd ..
cd ..

# file tree :
    # setup.sh
    # requirements.txt
    # hw4p2.py
    # data
    # --- full
    # --- toy
