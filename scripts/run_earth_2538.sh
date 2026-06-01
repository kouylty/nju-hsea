#!/bin/bash

seed_start=2022
seed_end=2025
epochs=5000
task=earth
task_name=earth_2538
dims=2538
pop_size=20
use_checkpoint=False
correlation_threshold=0.075
archive_size=20
ea_mutation_type='mix'
ea_selection_type='elite'
qd_mutation_type='mix'
init_sampler_type='init_v3'
# policy_path="multi_envs_models/dim2538_epoch2000.pth"
uni_policy_path="unified_envs_models/epoch2000_10_partial_dims.pth"
uni_policy_path_large="unified_envs_models_large/epoch2000_10_partial_dims.pth"
num_segments=32
device="cuda:0"

for ((seed=$seed_start; seed<=$seed_end; seed++))
do
    {
    # # map_elites
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=map_elites \
    #     algorithm.name=map_elites \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     algorithm.model.device=$device \
    #     use_checkpoint=$use_checkpoint
    
    # # RLEA
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=RLEA \
    #     algorithm.name=RLEA \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.policy_path=$policy_path \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # RLQD
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=RLQD \
    #     algorithm.name=RLQD \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.policy_path=$policy_path \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # # UniRLEA
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA \
    #     algorithm.name=UniRLEA \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.policy_path=$uni_policy_path \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD \
    #     algorithm.name=UniRLQD \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.policy_path=$uni_policy_path \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # UniRLEA_Large
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        algorithm=UniRLEA_Large \
        algorithm.name=UniRLEA_Large \
        seed=$seed \
        algorithm.model.pop_size=$pop_size\
        algorithm.model.num_segments=$num_segments \
        algorithm.model.policy_path=$uni_policy_path_large \
        algorithm.model.device=$device \
        algorithm.model.mutation_type=$ea_mutation_type \
        algorithm.model.selection_type=$ea_selection_type \
        algorithm.model.init_sampler_type=$init_sampler_type \
        use_checkpoint=$use_checkpoint

    # UniRLQD_Large
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        algorithm=UniRLQD_Large \
        algorithm.name=UniRLQD_Large \
        algorithm.model.pop_size=$pop_size\
        algorithm.model.num_segments=$num_segments \
        algorithm.model.policy_path=$uni_policy_path_large \
        algorithm.model.device=$device \
        algorithm.model.correlation_threshold=$correlation_threshold \
        algorithm.model.archive_size=$archive_size \
        algorithm.model.mutation_type=$qd_mutation_type \
        algorithm.model.init_sampler_type=$init_sampler_type \
        seed=$seed \
        use_checkpoint=$use_checkpoint

    # # sa
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=sa \
    #     algorithm.name=sa \
    #     seed=$seed \
    #     algorithm.pop_size=$pop_size\
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # ea
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=ea \
    #     algorithm.name=ea \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # Bandit_ea
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=bandit_ea \
    #     algorithm.name=bandit_ea \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # dqn
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=dqn \
    #     algorithm.name=dqn \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     use_checkpoint=$use_checkpoint

    # dropout sa
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=124 \
    #     algorithm.model.reset_freq=$epochs \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=100 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=50 \
    #     algorithm.model.reset_freq=2000 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=10 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_sa \
    #     algorithm.model.inner_opt_type=sa \
    #     algorithm.model.n_init=1 \
    #     algorithm.model.active_dims=20 \
    #     algorithm.model.reset_freq=2000 \
    #     +algorithm.model.decay=0.99 \
    #     +algorithm.model.T=100 \
    #     +algorithm.model.update_freq=10 \
    #     +algorithm.model.mutation_type=swap \
    #     seed=$seed

    # # dropout ea
    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=124 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=50 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # python main.py \
    #     epochs=$(($epochs/20)) \
    #     task=$task \
    #     task.name=$task_name \
    #     task.file_path=$file_path \
    #     algorithm=dropout_any \
    #     algorithm.name=dropout_ea \
    #     algorithm.model.inner_opt_type=ea \
    #     algorithm.model.n_init=20 \
    #     algorithm.model.active_dims=20 \
    #     algorithm.model.reset_freq=$(($epochs/20)) \
    #     +algorithm.model.pop_size=20 \
    #     +algorithm.model.init_sampler_type=permutation \
    #     +algorithm.model.mutation_type=swap \
    #     +algorithm.model.crossover_type=order \
    #     seed=$seed

    # bops
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=bo \
    #     algorithm.name=bops \
    #     algorithm.model.active_dims=$dims \
    #     algorithm.model.acqf_opt_type=ea \
    #     seed=$seed

    # dropout bo
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=bo \
    #     algorithm.name=dropout_bo \
    #     algorithm.model.active_dims=10 \
    #     algorithm.model.acqf_opt_type=ea \
    #     algorithm.model.fillin_type=best_pos \
    #     seed=$seed

    # dropout random
    }
done
