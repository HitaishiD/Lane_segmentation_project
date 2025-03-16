### To reproduce the same environment

1. Run the on-start.sh script (stored on your local computer)
2. Add the public key in ~/.ssh/id_rsa.pub in github
3. Clone the github repo
4. Setup the conda environment + download the dataset

launch ./setenv.sh



### Things to change but use this for the moment

1. activate the conda env 
2. run `python test_processor.py`
3. `mv testing_preprocessed_masks/ preprocessed_mask`