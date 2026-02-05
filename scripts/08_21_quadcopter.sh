nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_better_attacks_seed_0" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_0.out &


nohup
python -u main.py --problem quadcopter --reg_n_samples 300 --reg_weight 150.0 --critic_n_samples 10 --critic_max_n_steps 10 --gpu 2 --rho reg

python -u main.py --problem quadcopter --reg_n_samples 300 --reg_weight 50.0 --critic_n_samples 10 --critic_max_n_steps 10 --gpu 2 --rho reg



nohup python -u main.py --problem quadcopter --rho reg --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_reg_copter_reg_150" --gpu 1 &> h_reg_copter_reg_150.out &

nohup python -u main.py --problem quadcopter --rho reg --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 50.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_reg_copter_reg_50" --gpu 1 &> h_reg_copter_reg_50.out &

nohup python -u main.py --problem quadcopter --rho reg --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 300.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_reg_copter_reg_150" --gpu 2 &> h_reg_copter_reg_300.out &

nohup python -u main.py --problem quadcopter --rho sum --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_sum_copter_reg_150" --gpu 2 &> h_sum_copter_reg_150.out &

nohup python -u main.py --problem quadcopter --rho sum --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 50.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_sum_copter_reg_50" --gpu 3 &> h_sum_copter_reg_50.out &

nohup python -u main.py --problem quadcopter --rho sum --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 300.0 --objective_option "weighted_average" --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "h_sum_copter_reg_150" --gpu 3 &> h_sum_copter_reg_300.out &






python -u main.py --problem quadcopter --reg_n_samples 5 --reg_weight 0.0 --critic_n_samples 10 --critic_max_n_steps 10 --gpu 1 --rho sum

--reg_weight TUNE
--reg_n_samples 250
--reg_sampler random or random_inside
--reg_transform sigmoid


--reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0
--objective_option "weighted_average"
--phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe
--critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian
--random_seed 0 --affix "ESG_reg_speedup_better_attacks_seed_0" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_0.out &

--rho reg or sum
