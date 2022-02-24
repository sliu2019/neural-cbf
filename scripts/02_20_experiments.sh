
# DON'T FORGET TO INCLUDE THE FOLLOWING OPTIONS
# --h max or --h sum
# --pend_length 3.0 or --pend_length 1.5
# --box_ang_vel_limit 20
# 20 rad/s: 3 rev/s; pretty fast!

# --no_softplus_on_obj
# --reg_weight

# Server 4:

# Easier env, flat vs. iterated construction
#nohup python main.py --problem flying_inv_pend --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-3 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix easier_env_flat_beta --gpu 3 &> easier_env_flat_beta.out &
#
#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix easier_env_iterated_beta --gpu 3 &> easier_env_iterated_beta.out &
#
########
## Harder env, flat vs. iterated construction
#nohup python main.py --pend_length 1.5 --problem flying_inv_pend --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix harder_env_flat_beta --gpu 0 &> harder_env_flat_beta.out &
#
#nohup python main.py --pend_length 1.5 --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix harder_env_iterated_beta --gpu 0 &> harder_env_iterated_beta.out &

# Easier env, larger phi_init randomness, 2 seeds
#nohup python main.py --problem flying_inv_pend --h sum --phi_ci_init_range 1.0 --random_seed 0 --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-3 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix easier_flat_more_random_seed_0 --gpu 2 &> easier_flat_more_random_seed_0.out &
#
#nohup python main.py --problem flying_inv_pend --h sum --phi_ci_init_range 1.0 --random_seed 2 --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix easier_flat_more_random_seed_2 --gpu 2 &> easier_flat_more_random_seed_2.out &

######################
### Evening run

## Pend 3.0, iterated beta
nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix pend_3_beta_iterated --gpu 0 &> pend_3_beta_iterated.out &

## Longer pend (5.0), beta iterated
nohup python main.py --problem flying_inv_pend --pend_length 5.0 --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix pend_5_beta_iterated --gpu 1 &> pend_5_beta_iterated.out &

## Pend 3.0, iterated beta, 128-128 NN
nohup python main.py --problem flying_inv_pend --phi_nn_dimension 128-128 --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix pend_3_beta_iterated_nn_128_128 --gpu 3 &> pend_3_beta_iterated_nn_128_128.out &

## Longer pend (5.0), beta flat, 128-128 NN
nohup python main.py --problem flying_inv_pend --pend_length 5.0 --phi_nn_dimension 128-128 --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix pend_5_beta_flat_nn_128_128 --gpu 3 &> pend_5_beta_flat_nn_128_128.out &