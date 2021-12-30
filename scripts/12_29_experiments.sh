# Troubleshooting the weird double-parabola

# First test new regularization term
# May need to tune the weight again

# 64-64; 60 points; gradient averaging; AND NOW, the new regularization term

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 1 --affix 64_64_60pts_40weight_gdavg_newreg --reg_weight 40 --random_seed 1 &> 64_64_60pts_40weight_gdavg_newreg.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 2 --affix 64_64_60pts_20weight_gdavg_newreg --reg_weight 20 --random_seed 1 &> 64_64_60pts_20weight_gdavg_newreg.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 3 --affix 64_64_60pts_50weight_gdavg_newreg --reg_weight 50 --random_seed 1 &> 64_64_60pts_50weight_gdavg_newreg.out &

# different weights
# fixed weight, different seeds

# tanh instead of relu for different seeds
# tanh (if it's fine), then replace adam with 2 different LR schedules (to be tuned!)