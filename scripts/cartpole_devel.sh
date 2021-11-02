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
#
