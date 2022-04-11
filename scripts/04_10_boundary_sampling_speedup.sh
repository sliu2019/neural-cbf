
# Trying to use multiprocessing for boundary sampling

#python main.py --problem flying_inv_pend --phi_nn_dimension 32-32 --phi_nnl tanh-tanh-softplus --phi_include_xe --phi_nn_inputs euc --no_softplus_on_obj --train_attacker_n_samples 10 --train_attacker_max_n_steps 2 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4000 --reg_sampler random --reg_weight 10.0 --trainer_average_gradients --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method gpu_parallelized --affix debug

# 	parser.add_argument("--gradient_batch_warmstart_faster_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"])
#   parser.add_argument("--gradient_batch_warmstart_faster_sampling_method", type=str, default="uniform", choices=["uniform", "gaussian"])


python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --affix "debug" --gpu 1 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method gpu_parallelized