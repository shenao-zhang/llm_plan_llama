conda activate shenao
torchrun --nproc_per_node 4 main.py \
        --num_trials 10 \
        --num_envs 134 \
        --run_name "ours_run_logs" \
        --use_memory
