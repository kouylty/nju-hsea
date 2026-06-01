#!/bin/bash

seed_start=2022
seed_end=2022
epochs=2000
task=earth
task_name=earth_124
dims=124
pop_size=20
correlation_threshold=0.68
archive_size=20
use_checkpoint=False
policy_path="multi_envs_models/dim124_epoch2000.pth"
uni_policy_path="unified_envs_models/epoch2000_10_partial_dims.pth"
uni_policy_path_large="unified_envs_models_large/epoch2000_10_partial_dims.pth"
uni_policy_path_large_metadata="unified_envs_models_large_with_metadata/epoch2000_trainenvs200_10_partial_dims135_numsegment32_dmodel64.pth"
uni_policy_path_large_metadata_ngs="unified_envs_models_large_with_metadata_ngs/epoch2000_trainenvs200_10_partial_dims135_numsegment32_dmodel64.pth"
# uni_policy_path_large_metadata_ngs_withfullmask="unified_envs_models_large_with_metadata_ngs_withfullmask/epoch2000_trainenvs200_10_partial_dims135_numsegment32_dmodel64.pth"
uni_policy_path_large_metadata_ngs_withfullmask="unified_envs_models_large_with_metadata_ngs_withfullmask/epoch3000_trainenvs200_20_partial_dims135_numsegment32_dmodel64.pth"
uni_policy_path_large_metadata_ngs_withfullmask_window="unified_envs_models_large_with_metadata_ngs_withfullmask_window/epoch1000_trainenvs100_10_partial_dims135_numsegment32_dmodel64_window4.pth"


window_size=4
num_segments=32
ea_mutation_type='insert'
ea_selection_type='elite'
qd_mutation_type='adaptive_mix'
init_sampler_type='init_v3'
device="cpu"
adaptive_correlation=false
local_search=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mutation)
            qd_mutation_type="$2"
            shift 2
            ;;
        --adaptive-correlation)
            adaptive_correlation="$2"
            shift 2
            ;;
        --local-search)
            local_search="$2"
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
    {
    # map_elites
    python main.py \
        epochs=$epochs \
        task=$task \
        task.name=$task_name \
        algorithm=map_elites \
        algorithm.name=map_elites \
        seed=$seed \
        algorithm.model.pop_size=$pop_size\
        algorithm.model.correlation_threshold=$correlation_threshold \
        algorithm.model.archive_size=$archive_size \
        algorithm.model.mutation_type=$qd_mutation_type \
        algorithm.model.init_sampler_type=$init_sampler_type \
        algorithm.model.device=$device \
        mutation=$qd_mutation_type \
        adaptive_correlation=$adaptive_correlation \
        local_search.enabled=$local_search \
        local_search.freq=50 \
        local_search.top_k=1 \
        local_search.num_neighbors=4 \
        use_checkpoint=$use_checkpoint
    
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

    
    # # UniRLEA_Large
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA_Large \
    #     algorithm.name=UniRLEA_Large \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD_Large
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD_Large \
    #     algorithm.name=UniRLQD_Large \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # # UniRLEA_Large_Metadata
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA_Large_Metadata \
    #     algorithm.name=UniRLEA_Large_Metadata \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD_Large_Metadata
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD_Large_Metadata \
    #     algorithm.name=UniRLQD_Large_Metadata \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # # UniRLEA_Large_Metadata_ngs
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA_Large_Metadata_ngs \
    #     algorithm.name=UniRLEA_Large_Metadata_ngs \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD_Large_Metadata_ngs
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD_Large_Metadata_ngs \
    #     algorithm.name=UniRLQD_Large_Metadata_ngs \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # # UniRLEA_Large_Metadata_ngs_withfullmask
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA_Large_Metadata_ngs_withfullmask \
    #     algorithm.name=UniRLEA_Large_Metadata_ngs_withfullmask \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs_withfullmask \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD_Large_Metadata_ngs_withfullmask
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD_Large_Metadata_ngs_withfullmask \
    #     algorithm.name=UniRLQD_Large_Metadata_ngs_withfullmask \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs_withfullmask \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

    # # UniRLEA_Large_Metadata_ngs_withfullmask_window
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLEA_Large_Metadata_ngs_withfullmask_window \
    #     algorithm.name=UniRLEA_Large_Metadata_ngs_withfullmask_window \
    #     seed=$seed \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs_withfullmask_window \
    #     algorithm.model.device=$device \
    #     algorithm.model.mutation_type=$ea_mutation_type \
    #     algorithm.model.selection_type=$ea_selection_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     use_checkpoint=$use_checkpoint

    # # UniRLQD_Large_Metadata_ngs_withfullmask_window
    # python main.py \
    #     epochs=$epochs \
    #     task=$task \
    #     task.name=$task_name \
    #     algorithm=UniRLQD_Large_Metadata_ngs_withfullmask \
    #     algorithm.name=UniRLQD_Large_Metadata_ngs_withfullmask \
    #     algorithm.model.pop_size=$pop_size\
    #     algorithm.model.num_segments=$num_segments \
    #     algorithm.model.policy_path=$uni_policy_path_large_metadata_ngs_withfullmask \
    #     algorithm.model.device=$device \
    #     algorithm.model.correlation_threshold=$correlation_threshold \
    #     algorithm.model.archive_size=$archive_size \
    #     algorithm.model.mutation_type=$qd_mutation_type \
    #     algorithm.model.init_sampler_type=$init_sampler_type \
    #     seed=$seed \
    #     use_checkpoint=$use_checkpoint

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
