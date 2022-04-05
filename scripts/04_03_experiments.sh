
Experiments:

What's new? Including yaw (explicitly or implicitly in euc) or transform to euc for easier learning


For all, use format 2 or 1 (check?)

All angles, deeper net

Euclidean, 2 layer net

Try one of these with softplus at end


For timing:
Try
	parser.add_argument('--train_attacker', default='gradient_batch_warmstart', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart', 'gradient_batch_warmstart2'])
	# TODO: new below
		parser.add_argument('--gradient_batch_warmstart2_proj_tactic', choices=['gd_step_timeout', 'adam_ba'])

Use a more infrequent test metric (per 100 or 250 iterations) and when it exits

Add some reg too

--n_test_loss_step 100 # how often it's computed depends on how long it takes?
#******************************************************************
#
#Debug code: probably test one thing at a time

whether or not softplus is on the objective
				# TODO: the fact that this is not = self.args.n_checkpoint_step necessarily means that you might have to refactor stuff in flying_rollout_experiment


nohup python main.py --problem flying_inv_pend --phi_format 2 --phi_nn_inputs "angles_derivs_no_yaw" --phi_include_xe --phi_nn_dimension "32-32" --no_softplus_on_obj --train_attacker_n_samples 120 --train_attacker_max_n_steps 50 --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --random_seed 1 --affix phi_format_2_seed_1 --gpu 3  &> phi_format_2_seed_1.out &

# Debug main, phi_fn
python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --affix debug

# Debug trainer
python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --affix debug --no_softplus_on_obj --train_attacker_n_samples 10 --train_attacker_max_n_steps 2  --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler fixed --reg_weight 0.0 --trainer_average_gradients

# Try with reg: (check if grad norm is being computed)
# Data: do we see CPU clogging? Do we fill out the data correctly? (Examine)
python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --affix debug --no_softplus_on_obj --train_attacker_n_samples 10 --train_attacker_max_n_steps 2  --train_attacker_use_n_step_schedule --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler random --reg_weight 10.0 --trainer_average_gradients

# Testing test metric and logging
python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --affix debug --train_attacker_n_samples 10 --train_attacker_max_n_steps 2 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler random --reg_weight 10.0 --objective_option weighted_average


# Testing new attacker
python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --affix debug --no_softplus_on_obj --train_attacker_n_samples 10 --train_attacker_max_n_steps 2 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler random --reg_weight 10.0 --trainer_average_gradients --train_attacker gradient_batch_warmstart2 --gradient_batch_warmstart2_proj_tactic adam_ba


#	parser.add_argument('--train_attacker', default='gradient_batch_warmstart', choices=['basic', 'gradient_batch', 'gradient_batch_warmstart', 'gradient_batch_warmstart2'])
#	parser.add_argument('--gradient_batch_warmstart2_proj_tactic', choices=['gd_step_timeout', 'adam_ba'])


