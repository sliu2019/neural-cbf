# (Week of) 3/13: making changes to algorithm to make it train as desired

# First: test that the 3 new RegSampler classes work
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler random
#
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler fixed
#
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler boundary

# Run: some experiments to see if RandomRegSampler is effective

# tuning reg_weight
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 3 --reg_sampler fixed --reg_weight 0.0


# Notable arguments:
# Stronger counterexample generator
# We're mainly testing a different volume reg tactic.
# Smaller things we're testing:
# 1. Reg through smaller phi_nn_dimension
# 2. Counterexample effectiveness: more counterexamples, plus an exponential step schedule
# Hyperparam tuning: reg weight

# Server 4
# reg weight 100
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 100.0 --affix reg_weight_100 --gpu 3 &> reg_weight_100.out &

# Everything else: server 5
## reg weight 10
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 10.0 --affix reg_weight_10 --gpu 0 &> reg_weight_10.out &
#
## reg weight 50
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 50.0 --affix reg_weight_50 --gpu 1 &> reg_weight_50.out &
#
## reg weight 150
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 150.0 --affix reg_weight_150 --gpu 2 &> reg_weight_150.out &
#
## reg weight 200
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 200.0 --affix reg_weight_200 --gpu 3 &> reg_weight_200.out &

## reg weight 1
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 1.0 --affix reg_weight_1 --gpu 0 &> reg_weight_1.out &
#
## reg weight 0.1
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 0.1 --affix reg_weight_1e-1 --gpu 1 &> reg_weight_1e-1.out &


# reg weight 1, phi dim 64-64
nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 1.0 --affix reg_weight_1_phi_dim_64_64 --gpu 2 &> reg_weight_1_phi_dim_64_64.out &

# reg weight 0.1, phi dim 64-64
nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --reg_sampler random --reg_weight 0.1 --affix reg_weight_1e-1_phi_dim_64_64 --gpu 3 &> reg_weight_1e-1_phi_dim_64_64.out &
