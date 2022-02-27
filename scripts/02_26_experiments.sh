
# Debugging reg term
#python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 1.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4500 --affix debug --gpu 1

## ROUGH!!! Parameter search on reg weight
### YOU REALLY HAVE TO WATCH THESE FOR THE DEBUG INPUTS

#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 20.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_20 --gpu 0 &> reg_weight_20.out &
#
#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 45.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_45 --gpu 1 &> reg_weight_45.out &
#
#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 60.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_60 --gpu 2 &> reg_weight_60.out &
#
#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 100.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_100 --gpu 3 &> reg_weight_100.out &

#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 1.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_1 --gpu 0 &> reg_weight_1.out &

#nohup python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 0.1 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix reg_weight_1e-1 --gpu 1 &> reg_weight_1e-1.out &

### Sunday afternoon 02.27
### Debugging reg term

# Did I break it?
#python main.py --train_attacker_projection_time_limit 3.0 --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 1.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug_weight_1 --gpu 0
