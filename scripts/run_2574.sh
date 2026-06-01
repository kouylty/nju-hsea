#!/bin/bash

# This script runs the Evolutionary Algorithm (EA) on the earth_2574 dataset.

# Experiment settings
seed=2025
epochs=20000
task=earth
task_name=earth_2574
pop_size=20
use_checkpoint=False

echo "Running Evolutionary Algorithm (EA) on task ${task_name} with seed ${seed}"

python main.py \
    epochs=$epochs \
    task=$task \
    task.name=$task_name \
    algorithm=ea \
    algorithm.name=ea \
    seed=$seed \
    algorithm.model.pop_size=$pop_size\
    use_checkpoint=$use_checkpoint

echo "Run finished. Check the output in the 'data/earth_2574' directory." 