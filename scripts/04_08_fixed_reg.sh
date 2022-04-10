# Experiments after fixing reg
# Redo of 04-06 experiments

# All: Euc, tanh-tanh-softplus, gradient averaging
# Vary: reg format

# server 4
# Transform = sigmoid, regsampler = random_inside
# I think this should work, but below is backup. They will need different reg weights
nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 50.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "ESG_reg_sigmoid_random_inside_sampler_weight_50" --gpu 0 &> ESG_reg_sigmoid_random_inside_sampler_weight_50.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "ESG_reg_sigmoid_random_inside_sampler_weight_150" --gpu 0 &> ESG_reg_sigmoid_random_inside_sampler_weight_150.out &

nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "ESG_reg_sigmoid_random_inside_sampler_weight_200" --gpu 1 &> ESG_reg_sigmoid_random_inside_sampler_weight_200.out &

# Transform = sigmoid, regsampler = random_inside
# Backup for above


