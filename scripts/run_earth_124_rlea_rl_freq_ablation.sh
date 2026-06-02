#!/bin/bash

policy_path="multi_envs_models/dim124_epoch2000.pth"
epochs=2000
device="cuda:0"
seed=2022

rl_freqs=(
    500
    100
    50
    10
    1
)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --policy-path)
            policy_path="$2"
            shift 2
            ;;
        --epochs)
            epochs="$2"
            shift 2
            ;;
        --device)
            device="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --rl-freqs)
            IFS=',' read -r -a rl_freqs <<< "$2"
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
    echo "Train it first, or pass --policy-path to an existing checkpoint."
    exit 1
fi

for rl_freq in "${rl_freqs[@]}"
do
    echo "Running RLEA rl_freq=${rl_freq}"
    bash ./scripts/run_earth_124_rlea.sh \
        --policy-path "$policy_path" \
        --epochs "$epochs" \
        --device "$device" \
        --seed "$seed" \
        --rl-freq "$rl_freq"
done
