# S nonempty
#
#nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0 --affix exp1a --n_checkpoint_step 30 --gpu 0 &> exp1a.out &
#nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0.025 --affix exp1b --n_checkpoint_step 30 --gpu 1 &> exp1b.out &
##nohup python -u main.py --problem cartpole_reduced --physical_difficulty hard --objective_volume_weight 0 --affix exp1c --n_checkpoint_step 30 --gpu 2 &> exp1c.out &
#
##- Exp 1a: 2 vars, weight = 0, easy physical parameters
##- Exp 1b: 2 vars, weight = 0.025, easy physical parameters
##- Exp 1c: 2 vars, weight = 0, hard physical parameters
#
## python -u main.py --problem cartpole_reduced --affix debug
#python main.py --affix debug --gpu 2
#
#python -u main.py --affix exp1a --n_checkpoint_step 3 --train_attacker_projection_stop_threshold 1e-1 --test_attacker_projection_stop_threshold 1e-1 --train_attacker_projection_lr 1.0 --test_attacker_projection_lr 1.0 --trainer_early_stopping_patience 10
#
#python -u main.py --affix exp1a --n_checkpoint_step 1 --trainer_early_stopping_patience 10 --train_attacker_projection_stop_threshold 1e-2 --test_attacker_projection_stop_threshold 1e-2
#
## Scratch
#python main.py --affix devel --gpu 2 --phi_nn_dimension 50
#
#python main.py --affix devel --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 5e-4
#
## here
#python main.py --affix devel --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 1.0

# Simin: monday night
#nohup python main.py --affix l_50_w_1 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 &> l_50_w_1.out &
#nohup python main.py --affix l_50_w_5e_1 --gpu 2 --phi_nn_dimension 50 --objective_volume_weight 0.5 &> l_50_w_5e_1.out &
#nohup python main.py --affix l_50_w_1e_1 --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 0.1 &> l_50_w_1e_1.out &

#python main.py --affix l_50_50_w_1e_2 --gpu 2 --phi_nn_dimension 50-50
#nohup python main.py --affix l_50_50_w_1e_2 --gpu 2 --phi_nn_dimension 50-50 --objective_volume_weight 1e-2 &> l_50_50_w_1e_2.out &
#nohup python main.py --affix l_50_50_w_1e_4 --gpu 3 --phi_nn_dimension 50-50 --objective_volume_weight 1e-4 &> l_50_50_w_1e_4 &
#nohup python main.py --affix l_50_50_w_5e_4 --gpu 1 --phi_nn_dimension 50-50 --objective_volume_weight 5e-4 &> l_50_50_w_5e_4.out &


# Rerun with different h(x)
# --phi_ci_init_range 1.0

#nohup python main.py --affix new_h_l_50_w_1 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 --trainer_early_stopping_patience 40 &> new_h_l_50_w_1.out &
#nohup python main.py --affix new_h_l_50_w_10 --gpu 2 --phi_nn_dimension 50 --objective_volume_weight 10.0 --trainer_early_stopping_patience 40 &> new_h_l_50_w_10.out &
#
#
#nohup python main.py --affix new_h_l_50_w_0 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 0.0 --trainer_early_stopping_patience 40 &> new_h_l_50_w_0.out &
#
#
#python main.py --affix debug --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 0.0

# After fixing a bunch of stuff with the attacks

#nohup python main.py --affix fixed_attacks_1 --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 &> fixed_attacks_1.out &

# Trying a bunch of random seeds
#nohup python main.py --affix r1 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 1 &> r1.out &
#nohup python main.py --affix r2 --gpu 2 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 2 &> r2.out &
#nohup python main.py --affix r3 --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 3 &> r3.out &
#nohup python main.py --affix r4 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 4 &> r4.out &
#nohup python main.py --affix r5 --gpu 2 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 50 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 5 &> r5.out &



#nohup python main.py --affix r101 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 10 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 101 --trainer_stopping_condition n_steps --trainer_n_steps 20000 &> r101.out &
#nohup python main.py --affix r102 --gpu 2 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 10 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 102 --trainer_stopping_condition n_steps --trainer_n_steps 20000 &> r102.out &
#nohup python main.py --affix r103 --gpu 3 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 10 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 103 --trainer_stopping_condition n_steps --trainer_n_steps 20000 &> r103.out &
#nohup python main.py --affix r104 --gpu 0 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 10 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 104 --trainer_stopping_condition n_steps --trainer_n_steps 20000 &> r104.out &
#nohup python main.py --affix r105 --gpu 1 --phi_nn_dimension 50 --objective_volume_weight 1.0 --n_checkpoint_step 10 --train_attacker_n_samples 50 --test_attacker_n_samples 50 --random_seed 105 --trainer_stopping_condition n_steps --trainer_n_steps 20000 &> r105.out &


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
python main.py --problem cartpole_reduced --affix debug_warmstart --train_attacker gradient_batch_warmstart --test_attacker gradient_batch_warmstart --n_checkpoint_step 1

python -u main.py --problem cartpole_reduced --affix debug

