
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
