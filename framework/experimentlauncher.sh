#! /bin/bash


## Experiment 1
python trainer.py --batch_size 5 --epochs 11 &

## Experiment 2
python trainer.py --batch_size 5 --epochs 14 &

## Experiment 3
python trainer.py --batch_size 5 --epochs 16 &


wait 

echo "all experiments are done"