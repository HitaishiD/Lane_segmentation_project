### To do immediately after starting the machine
Run the on-start.sh script (stored on your local computer)

1. `touch on-start.sh`
2. `vim on-start.sh`
3. copy the contents of on-start.sh
4. `./on-start.sh`


### To reproduce the same conda environment

1. Run the on-start.sh script (stored on your local computer)
2. Add the public key in ~/.ssh/id_rsa.pub in github
3. Clone the github repo
4. Setup the conda environment + download the dataset

launch `./setenv.sh`



### Things to change but use this for the moment

1. activate the conda env 
2. run `python test_processor.py`
3. `mv testing_preprocessed_masks/ preprocessed_mask`


### Running the training loop

The training loop can be run with different hyperparameters as shown
`python trainer.py --batch_size 5 --epochs 13`


### Plotting the results
For the moment the training loss file needs to be hard coded. but this can be changed later. A temporary solution run 


`python plotter.py`



### Experiment launcher
This is to launch several experiments in one script only

`./experimentlauncher.sh`

