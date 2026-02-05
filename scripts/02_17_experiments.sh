# Round 2 of experiments

nohup python main.py --problem flying_inv_pend --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix iterated_beta --gpu 0 &> iterated_beta.out &

nohup python main.py --problem flying_inv_pend --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix flat_beta --gpu 0 &> flat_beta.out &


nohup python main.py --problem flying_inv_pend --pend_length 9.0 --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 60 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix iterated_beta_pend_len_9 --gpu 1 &> iterated_beta_pend_len_9.out &

nohup python main.py --problem flying_inv_pend --pend_length 9.0 --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 60 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix flat_beta_pend_len_9 --gpu 1 &> flat_beta_pend_len_9.out &

#python main.py --problem flying_inv_pend --phi_ci_init_range 1 --random_seed 0 --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix iterated_beta_high_var_0 --gpu 1
#
#python main.py --problem flying_inv_pend --phi_ci_init_range 1 --random_seed 1 --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix iterated_beta_high_var_1 --gpu 1
#
#python main.py --problem flying_inv_pend --phi_ci_init_range 10 --random_seed 3 --no_softplus_on_obj --reg_weight 0.0 --phi_include_beta_deriv --phi_include_xe --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --affix iterated_beta_high_var_2 --gpu 1

#python main.py --no_softplus_on_obj --reg_weight 0.0 --phi_include_xe --phi_include_beta_deriv --phi_ci_init_range 10 --random_seed 0
#python main.py --no_softplus_on_obj --reg_weight 0.0 --phi_ci_init_range 10 --random_seed 0



#nohup python main.py --problem flying_inv_pend --phi_include_xe --phi_include_beta_deriv --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 30 --critic_projection_lr 1e-2 --learner_stopping_condition n_steps --learner_n_steps 3000 --reg_weight 0 --affix first_run_no_reg &> first_run_no_reg.out &
