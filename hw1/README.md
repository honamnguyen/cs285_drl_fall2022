## Question 1.2
Commands to produce results in Table 1
### Ants
```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant_evalbatch5000 --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1 --eval_batch_size 5000
```
```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant_evalbatch10000 --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1 --eval_batch_size 10000
```

### Walker2d
```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name bc_walker2d_evalbatch5000 --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
	--video_log_freq -1 --eval_batch_size 5000
```


## Question 1.3
Command to produce results in Figure 1 (run_logs not included as directed in HW1 PDF). This is specifically for `num_agent_train_steps_per_iter = 90`

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --n_iter 1 \
    --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 --eval_batch_size 5000 --num_agent_train_steps_per_iter 90 --exp_name bc_ant_trainsteps90
```

## Question 2.2
Commands to produce results in Figure 2
### Ant

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant_evalbatch5000 --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1 --eval_batch_size 5000
```

### Walker2d

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Walker2d.pkl \
    --env_name Walker2d-v4 --exp_name dagger_walker2d_evalbatch5000 --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
	--video_log_freq -1 --eval_batch_size 5000
```