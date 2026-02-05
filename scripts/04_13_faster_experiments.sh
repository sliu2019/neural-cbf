
# Redo Monday 4/10 experiments, which were too slow
# Server 4
# Changes:
#1. Decreased test_N_boundary_samples, test_N_volume_samples from 5000 to 2500
#2. Test critic is the fast version as well
#3. Timing test metric
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_weight_150_seed_0_again" --gpu 0 &> ESG_reg_speedup_weight_150_seed_0_again.out &
#
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 1 --affix "ESG_reg_speedup_weight_150_seed_1_again" --gpu 0 &> ESG_reg_speedup_weight_150_seed_1_again.out &
#
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 2 --affix "ESG_reg_speedup_weight_150_seed_2_again" --gpu 1 &> ESG_reg_speedup_weight_150_seed_2_again.out &
#
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 3 --affix "ESG_reg_speedup_weight_150_seed_3_again" --gpu 1 &> ESG_reg_speedup_weight_150_seed_3_again.out &
#
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 4 --affix "ESG_reg_speedup_weight_150_seed_4_again" --gpu 2 &> ESG_reg_speedup_weight_150_seed_4_again.out &

###########################
# Server 5
#Changes:
#1. 5x the number of counterexamples
#2. Fewer counterexample opt steps: 50 --> 20
#3. p_reuse increase from 0.3 --> 0.5

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_better_attacks_seed_0" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_0.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 1 --affix "ESG_reg_speedup_better_attacks_seed_1" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_1.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 2 --affix "ESG_reg_speedup_better_attacks_seed_2" --gpu 1 &> ESG_reg_speedup_better_attacks_seed_2.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 3 --affix "ESG_reg_speedup_better_attacks_seed_3" --gpu 2 &> ESG_reg_speedup_better_attacks_seed_3.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --critic gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 4 --affix "ESG_reg_speedup_better_attacks_seed_4" --gpu 3 &> ESG_reg_speedup_better_attacks_seed_4.out &