# Experiments after fixing reg
# Redo of 04-06 experiments
# Sped-up reg

# All: Euc, tanh-tanh-softplus, gradient averaging + speedup
# Vary: reg format

# server 4
# Transform = sigmoid, regsampler = random_inside
# I think this should work, but below is backup. They will need different reg weights

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_weight_150_seed_0" --gpu 0 &> ESG_reg_speedup_weight_150_seed_0.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 1 --affix "ESG_reg_speedup_weight_150_seed_1" --gpu 0 &> ESG_reg_speedup_weight_150_seed_1.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 2 --affix "ESG_reg_speedup_weight_150_seed_2" --gpu 1 &> ESG_reg_speedup_weight_150_seed_2.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 3 --affix "ESG_reg_speedup_weight_150_seed_3" --gpu 1 &> ESG_reg_speedup_weight_150_seed_3.out &

###########################
#### Batch 2

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 125.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_weight_125_seed_0" --gpu 2 &> ESG_reg_speedup_weight_125_seed_0.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 125.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 1 --affix "ESG_reg_speedup_weight_125_seed_1" --gpu 2 &> ESG_reg_speedup_weight_125_seed_1.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 125.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 2 --affix "ESG_reg_speedup_weight_125_seed_2" --gpu 3 &> ESG_reg_speedup_weight_125_seed_2.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 125.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 3 --affix "ESG_reg_speedup_weight_125_seed_3" --gpu 3 &> ESG_reg_speedup_weight_125_seed_3.out &



