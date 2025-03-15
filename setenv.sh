# set up conda environment

conda env create -f env.yml

sleep 5

conda activate cv


sleep 5
## Download dataset

sudo ./datasetdownload.sh
