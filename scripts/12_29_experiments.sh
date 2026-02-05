# Troubleshooting the weird double-parabola

# First test new regularization term
# May need to tune the weight again

# 64-64; 60 points; gradient averaging; AND NOW, the new regularization term

#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 1 --affix 64_64_60pts_40weight_gdavg_newreg --reg_weight 40 --random_seed 1 &> 64_64_60pts_40weight_gdavg_newreg.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 2 --affix 64_64_60pts_20weight_gdavg_newreg --reg_weight 20 --random_seed 1 &> 64_64_60pts_20weight_gdavg_newreg.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 3 --affix 64_64_60pts_50weight_gdavg_newreg --reg_weight 50 --random_seed 1 &> 64_64_60pts_50weight_gdavg_newreg.out &

###########################################################################################
# Test + tune new reg term: weights at different scales
# Do we successfully avoid the double parabola? (If we stop at the right iteration?)

#nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_min 2.5 --phi_a_init_max 7.5 --reg_sample_distance 0.2 --reg_weight 10 --gpu 0 --affix reg_point3_sigmoid_regweight_10 &> reg_point3_sigmoid_regweight_10.out &
#
#nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_min 2.5 --phi_a_init_max 7.5 --reg_sample_distance 0.2 --reg_weight 50 --gpu 1 --affix reg_point3_sigmoid_regweight_50 &> reg_point3_sigmoid_regweight_50.out &
#
#nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_min 2.5 --phi_a_init_max 7.5 --reg_sample_distance 0.2 --reg_weight 100 --gpu 2 --affix reg_point3_sigmoid_regweight_100 &> reg_point3_sigmoid_regweight_100.out &
#
#nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_min 2.5 --phi_a_init_max 7.5 --reg_sample_distance 0.2 --reg_weight 250 --gpu 3 --affix reg_point3_sigmoid_regweight_250 &> reg_point3_sigmoid_regweight_250.out &

# --reg_sample_distance 0.2
# --reg_weight: 10, 50, 100, 250
# --phi_a_init_min 2.5(/0.6 denom)
# --phi_a_init_min 7.5(/0.6 denom)
# --phi_ci_init_range 0.1
# regular settings: 64-64, 60 samples, no gradient averaging

######################################
# --reg_sample_distance 0.1 (default): do the smallest you can afford
# --reg_weight: 250, 500, 750
# --phi_a_init_max 0.01
# --phi_ci_init_range 0.01
# regular settings: 64-64, 60 samples, no gradient averaging

nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_max 1e-2 --reg_weight 250 --gpu 1 --affix reg_point3_sigmoid_regweight_250_init_small &> reg_point3_sigmoid_regweight_250_init_small.out &

nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_max 1e-2 --reg_weight 500 --gpu 2 --affix reg_point3_sigmoid_regweight_500_init_small &> reg_point3_sigmoid_regweight_500_init_small.out &

nohup python main.py --critic gradient_batch_warmstart --phi_nn_dimension 64-64 --critic_n_samples 60 --critic_max_n_steps 50 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_a_init_max 1e-2 --reg_weight 750 --gpu 3 --affix reg_point3_sigmoid_regweight_750_init_small &> reg_point3_sigmoid_regweight_750_init_small.out &

# Robustness to randomness: fixed weight, different seeds
# Robustness to initialization bias? Maybe it's not necessary

# Small fix: tanh instead of relu for different seeds
# Fix to help convergence: adam with 2 different LR schedules (to be tuned!)