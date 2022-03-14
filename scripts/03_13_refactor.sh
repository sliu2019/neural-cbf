# (Week of) 3/13: making changes to algorithm to make it train as desired

# First: test that the 3 new RegSampler classes work
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler random
#
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler fixed
#
#python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler boundary

# Run: some experiments to see if RandomRegSampler is effective

# Notable arguments:
# Stronger counterexample generator
python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 120 --train_attacker_max_n_steps 150 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler random --reg_weight 1.0

python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_n_samples 120 --train_attacker_max_n_steps 150 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 0 --reg_sampler random --reg_weight 1.0


--phi_nn_dimension "32-32"

--random_seed


# tuning reg_weight
python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --affix debug --gpu 3 --reg_sampler fixed --reg_weight 0.0
