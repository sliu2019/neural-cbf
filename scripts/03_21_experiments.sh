
# Description of runs: see if loss can decrease past 0 without regularization; if other CBF designs train better; and one run to debug reg gradient


# See if objective can decrease past 0.0
# Hint: FixedRegSampler will make things slightly faster
#nohup python main.py --problem flying_inv_pend --phi_include_beta_deriv --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler random --reg_weight 250.0 --reg_n_samples 500 --random_seed 0 --affix reg_weight_250_reg_n_samples_500_no_softplus_on_obj_seed_0 --gpu 0  &> reg_weight_250_reg_n_samples_500_no_softplus_on_obj_seed_0.out &


# Testing out CBF designs

# Design 0
python main.py --problem flying_inv_pend --phi_format 0 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug

# Design 1
python main.py --problem flying_inv_pend --phi_format 1 --phi_nn_inputs "angles_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug

# Design 2
python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug