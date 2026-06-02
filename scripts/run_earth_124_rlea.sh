#!/bin/bash

seed_start=2022
seed_end=2022
epochs=2000
task=earth
task_name=earth_124
pop_size=20
use_checkpoint=False
policy_path="multi_envs_models/dim124_epoch2000.pth"
init_sampler_type='init_v3'
mutation_type='insert'
crossover_type='pmx'
parent_selection_type='tournament'
tournament_size=3
selection_type='rank_based_prioritized'
device="cuda:0"
rl_freq=500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)
            policy_path="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --mutation)
            mutation_type="$2"
            shift 2
            ;;
        --crossover)
            crossover_type="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --rl-freq)
            rl_freq="$2"
            shift 2
            ;;
        --seed)
            seed_start="$2"
            seed_end="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ ! -f "$policy_path" ]]; then
    echo "Policy checkpoint not found: $policy_path"
    echo "Train it first, for example:"
    echo "python train.py --task_name EarthBenchEnv_124 --epoch 2000 --train_envs 100 --test_envs 100 --save_path $policy_path --device cuda:0"
    exit 1
fi

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    echo "Running RLEA earth_124 seed=${seed}, policy=${policy_path}, device=${device}, rl_freq=${rl_freq}"
    python main.py \
        epochs=$epochs \
        rl_freq=$rl_freq \
        task=$task \
        task.name=$task_name \
        algorithm=RLEA \
        algorithm.name=RLEA \
        seed=$seed \
        algorithm.model.pop_size=$pop_size \
        algorithm.model.policy_path=$policy_path \
        algorithm.model.device=$device \
        algorithm.model.mutation_type=$mutation_type \
        algorithm.model.crossover_type=$crossover_type \
        algorithm.model.parent_selection_type=$parent_selection_type \
        algorithm.model.tournament_size=$tournament_size \
        algorithm.model.selection_type=$selection_type \
        algorithm.model.init_sampler_type=$init_sampler_type \
        local_search.enabled=false \
        use_checkpoint=$use_checkpoint
done
