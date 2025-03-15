#! /bin/bash

mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Wait for 5 seconds
sleep 5

export PATH="$HOME/miniconda3/bin:$PATH"
conda init


# Wait for 5 seconds
sleep 5
## Keygen 

ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""




