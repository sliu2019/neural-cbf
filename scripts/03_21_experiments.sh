
# Testing out CBF designs for compilation

## Design 0
#python main.py --problem flying_inv_pend --phi_format 0 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug
#
## Design 1
#python main.py --problem flying_inv_pend --phi_format 1 --phi_nn_inputs "angles_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug
#
## Design 2
#python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --affix debug


# Testing them for training effectiveness (across 2 seeds), no reg

# Design 0
nohup python main.py --problem flying_inv_pend --phi_format 0 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 0 --affix phi_format_0_seed_0 --gpu 0  &> phi_format_0_seed_0.out &

nohup python main.py --problem flying_inv_pend --phi_format 0 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 1 --affix phi_format_0_seed_1 --gpu 3  &> phi_format_0_seed_1.out &

# Design 1
nohup python main.py --problem flying_inv_pend --phi_format 1 --phi_nn_inputs "angles_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 0 --affix phi_format_1_seed_0 --gpu 1  &> phi_format_1_seed_0.out &

nohup python main.py --problem flying_inv_pend --phi_format 1 --phi_nn_inputs "angles_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 1 --affix phi_format_1_seed_1 --gpu 1  &> phi_format_1_seed_1.out &

# Design 2
nohup python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 0 --affix phi_format_2_seed_0 --gpu 2  &> phi_format_2_seed_0.out &

nohup python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 1 --affix phi_format_2_seed_1 --gpu 3  &> phi_format_2_seed_1.out &


# Debug
python main.py --problem flying_inv_pend --phi_format 0 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 0 --affix debug --gpu 1

python main.py --problem flying_inv_pend --phi_format 1 --phi_nn_inputs "angles_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 0 --affix debug --gpu 1

python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --critic_n_samples 120 --critic_max_n_steps 50 --critic_use_n_step_schedule --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 1 --affix debug --gpu 1
