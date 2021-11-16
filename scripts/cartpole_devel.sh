# 11/09: after small fixes
#python main.py --affix debug --gpu 2

# 11/09
# Running the necessary experiments
# May run out of mem (saving frequently)

# Baseline: easier physical problem liits
#python main.py --affix baseline --gpu 1 --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500
## dS
#python main.py --affix dS --gpu 2 --train_mode dS --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500
#
## Easier physical problem limits
#python main.py --affix easier --gpu 3 --max_force 50.0 --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500
#
## xy only
#python main.py --affix xyonly --gpu 0 --g_input_is_xy --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500

nohup python main.py --affix baseline --gpu 1 --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500 &> baseline.out &
# dS
nohup python main.py --affix dS --gpu 2 --train_mode dS --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500 &> dS.out &

# Easier physical problem limits
nohup python main.py --affix easier --gpu 3 --max_force 50.0 --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500 &> easier.out &

# xy only
nohup python main.py --affix xyonly --gpu 0 --g_input_is_xy --phi_nn_dimension 16 --n_checkpoint_step 1 --trainer_stopping_condition n_steps --trainer_n_steps 1500 &> xyonly.out &

# Weekend of 11/14
# Debugging warmstart
# Used for finding warmstart parameters
python main.py --affix debug_warmstart --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --test_attacker gradient_batch --test_attacker_n_samples 50 --n_model_checkpoint_step 1 --n_data_checkpoint_step 1

#python -u main.py --problem cartpole_reduced --affix debug

# Monday 11/15
python main.py --affix throwaway --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --test_attacker gradient_batch --test_attacker_n_samples 50 --n_model_checkpoint_step 1 --n_data_checkpoint_step 1
