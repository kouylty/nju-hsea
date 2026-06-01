#!/bin/bash

seed_start=2022
seed_end=2023
epochs=50000
task=tsp
dataset=pcb1173
task_name=tsp_$dataset
file_path=benchmarks/tsp_data/$dataset
dims=1173

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    # dropout sa
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        task.file_path=$file_path \
        algorithm=dropout_any \
        algorithm.name=dropout_sa \
        algorithm.model.inner_opt_type=sa \
        algorithm.model.n_init=1 \
        algorithm.model.active_dims=1173 \
        algorithm.model.reset_freq=$epochs \
        +algorithm.model.decay=0.99 \
        +algorithm.model.T=100 \
        +algorithm.model.update_freq=100 \
        +algorithm.model.mutation_type=swap \
        seed=$seed

    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        task.file_path=$file_path \
        algorithm=dropout_any \
        algorithm.name=dropout_sa \
        algorithm.model.inner_opt_type=sa \
        algorithm.model.n_init=1 \
        algorithm.model.active_dims=700 \
        algorithm.model.reset_freq=200 \
        +algorithm.model.decay=0.99 \
        +algorithm.model.T=100 \
        +algorithm.model.update_freq=10 \
        +algorithm.model.mutation_type=swap \
        seed=$seed

    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        task.file_path=$file_path \
        algorithm=dropout_any \
        algorithm.name=dropout_sa \
        algorithm.model.inner_opt_type=sa \
        algorithm.model.n_init=1 \
        algorithm.model.active_dims=500 \
        algorithm.model.reset_freq=200 \
        +algorithm.model.decay=0.99 \
        +algorithm.model.T=100 \
        +algorithm.model.update_freq=10 \
        +algorithm.model.mutation_type=swap \
        seed=$seed

    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        task.file_path=$file_path \
        algorithm=dropout_any \
        algorithm.name=dropout_sa \
        algorithm.model.inner_opt_type=sa \
        algorithm.model.n_init=1 \
        algorithm.model.active_dims=300 \
        algorithm.model.reset_freq=200 \
        +algorithm.model.decay=0.99 \
        +algorithm.model.T=100 \
        +algorithm.model.update_freq=10 \
        +algorithm.model.mutation_type=swap \
        seed=$seed
    }
done