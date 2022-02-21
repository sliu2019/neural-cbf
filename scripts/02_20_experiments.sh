
# DON'T FORGET TO INCLUDE THE FOLLOWING OPTIONS
# --h max or --h sum
# --pend_length 3.0 or --pend_length 1.5
# --box_ang_vel_limit 20
# 20 rad/s: 3 rev/s; pretty fast!

# --no_softplus_on_obj
# --reg_weight

# Server 4: run up to 8
nohup python main.py --problem flying_inv_pend --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-3 --trainer_stopping_condition n_steps --trainer_n_steps 2000 --affix easier_env_flat_beta --gpu 0 &> easier_env_flat_beta.out &

nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 2000 --affix easier_env_iterated_beta --gpu 0 &> easier_env_iterated_beta.out &

#######
nohup python main.py --pend_length 1.5 --problem flying_inv_pend --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 2000 --affix harder_env_flat_beta --gpu 1 &> harder_env_flat_beta.out &

nohup python main.py --pend_length 1.5 --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 2000 --affix harder_env_iterated_beta --gpu 1 &> harder_env_iterated_beta.out &

python main.py --pend_length 1.5 --problem flying_inv_pend --h sum --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 2000 --affix harder_env_flat_beta --gpu 2