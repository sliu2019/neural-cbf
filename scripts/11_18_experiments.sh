# Baseline
nohup python main.py --affix 11_18_baseline --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --gpu 0 &> 11_18_baseline.out &

# More parameters in NN: does more ever hurt?
# --phi_nn_dimension 64-64-64
nohup python main.py --affix 64_64 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --gpu 1 &> 64_64.out &

# --phi_nn_dimension 64-64
nohup python main.py --affix 64_64_64 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64-64 --gpu 2 &> 64_64_64.out &

# --trainer_average_gradients
nohup python main.py --affix 64_64_gradient_avging --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 3 &> 64_64_gradient_avging.out &

# --g_input_is_xy
nohup python main.py --affix 64_64_xy --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --g_input_is_xy --gpu 0 &> 64_64_xy.out &

# --train_mode dS
# Not running. Thought about it, probably won't make a difference
#python main.py --affix 64_64_dS --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --train_mode dS


