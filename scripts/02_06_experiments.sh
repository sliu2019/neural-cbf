#python main.py --problem flying_inv_pend --phi_include_xe --phi_include_beta_deriv --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 2


# Testing projection with larger LR
# Will it work at all? Even if so, reg weight is bound to be way off...
#python main.py --problem flying_inv_pend --phi_include_xe --phi_include_beta_deriv --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_n_steps 1500 --reg_weight 40 --affix first_run &> first_run.out &

#nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --phi_nn_dimension 64-64 --gpu 0 --affix debugpinch1 --reg_weight 40 --random_seed 1 --phi_k0_init_max 10 &> debugpinch1.out &

# Remove reg weight and see what happens
#nohup python main.py --problem flying_inv_pend --phi_include_xe --phi_include_beta_deriv --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --reg_weight 0 --affix first_run_no_reg &> first_run_no_reg.out &


# Long pend, larger state box
nohup python main.py --problem flying_inv_pend --phi_include_xe --phi_include_beta_deriv --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 30 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix long_pend &> long_pend.out &
