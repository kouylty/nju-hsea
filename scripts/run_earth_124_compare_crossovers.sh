#!/bin/bash

seed_start=2022
seed_end=2022
epochs=2000
task=earth
task_name=earth_124
pop_size=20
correlation_threshold=0.68
archive_size=20
use_checkpoint=False
init_sampler_type='init_v3'
device="cpu"
mutation_type='mix'

crossovers=(
    order
    pmx
    cycle
)

for crossover_type in "${crossovers[@]}"
do
    for ((seed=$seed_start; seed<=$seed_end; seed++))
    do
        echo "Running earth_124 map_elites mutation=${mutation_type}, crossover=${crossover_type}, seed=${seed}"
        python main.py \
            epochs=$epochs \
            task=$task \
            task.name=$task_name \
            algorithm=map_elites \
            algorithm.name=map_elites \
            seed=$seed \
            algorithm.model.pop_size=$pop_size \
            algorithm.model.correlation_threshold=$correlation_threshold \
            algorithm.model.archive_size=$archive_size \
            algorithm.model.selection_type=tournament \
            algorithm.model.tournament_size=3 \
            algorithm.model.mutation_type=$mutation_type \
            algorithm.model.crossover_type=$crossover_type \
            algorithm.model.init_sampler_type=$init_sampler_type \
            algorithm.model.device=$device \
            mutation=$mutation_type \
            adaptive_correlation=false \
            algorithm.model.dynamic_correlation_threshold=false \
            algorithm.model.lns_enabled=false \
            algorithm.model.threshold_accepting_enabled=false \
            algorithm.model.diversity_bonus_enabled=false \
            local_search.enabled=false \
            com=true \
            use_checkpoint=$use_checkpoint
    done
done
