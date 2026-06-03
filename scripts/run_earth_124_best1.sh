#!/bin/bash

seed_start=2022
seed_end=2022
epochs=2000
task=earth
task_name=earth_124
pop_size=20
correlation_threshold=0.68
archive_size=20
init_sampler_type='init_v3'
device="cpu"
use_checkpoint=False
com=false

phase1_end=500
phase2_end=1500
local_search_freq=50
local_search_top_k=1
local_search_num_neighbors=4

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --seed)
            seed_start="$2"
            seed_end="$2"
            shift 2
            ;;
        --phase1-end)
            phase1_end="$2"
            shift 2
            ;;
        --phase2-end)
            phase2_end="$2"
            shift 2
            ;;
        --local-search-freq)
            local_search_freq="$2"
            shift 2
            ;;
        --com)
            com="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    echo "Running phased best1 earth_124 seed=${seed}: constraint <= ${phase1_end}, LNS <= ${phase2_end}, local search after ${phase2_end}"
    python main_best.py \
        epochs=$epochs \
        +best_phase1_end=$phase1_end \
        +best_phase2_end=$phase2_end \
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
        algorithm.model.crossover_type=cycle \
        algorithm.model.init_sampler_type=$init_sampler_type \
        algorithm.model.device=$device \
        algorithm.model.constraint_repair_passes=2 \
        algorithm.model.lns_probability=0.1 \
        algorithm.model.lns_stagnation_epochs=100 \
        algorithm.model.lns_stagnation_probability=0.4 \
        local_search.enabled=true \
        local_search.freq=$local_search_freq \
        local_search.top_k=$local_search_top_k \
        local_search.num_neighbors=$local_search_num_neighbors \
        com=$com \
        use_checkpoint=$use_checkpoint
done
