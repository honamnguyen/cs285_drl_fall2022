#!bin/bash -x
eval "$(conda shell.bash hook)"
conda activate cs285

# for b in 100 200 300 400 500
# for seed in 0 #2 3 4 5 6 7 8 9
# for b in 10000 30000 50000 # q4_search
for lambda in 0 0.95 0.98 0.99 1 #q5 search
do
    # for r in 6e-3 7e-3 8e-3 9e-3 1e-2 2e-2 3e-2 4e-2 5e-2
    # for r in 0.005 0.01 0.02  # q4_search
    for r in 0.001
    do	
        # python cs285/scripts/run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b 1000 -lr 0.01 --seed $seed -rtg --exp_name q2_b1000_r0.01
        # python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 --discount 0.95 -n 100 -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline --exp_name q4_search_b${b}_lr${r}_rtg_nnbaseline  # q4_search
        python cs285/scripts/run_hw2.py --env_name Hopper-v4 --ep_len 1000 --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda $lambda --exp_name q5_b2000_r0.001_lambda$lambda # q5_search
    done
done
