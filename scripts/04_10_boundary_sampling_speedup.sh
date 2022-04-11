
# Trying to use multiprocessing for boundary sampling

# 	parser.add_argument("--gradient_batch_warmstart_faster_speedup_method", type=str, default="sequential", choices=["sequential", "gpu_parallelized", "cpu_parallelized"])
#   parser.add_argument("--gradient_batch_warmstart_faster_sampling_method", type=str, default="uniform", choices=["uniform", "gaussian"])
#   parser.add_argument("--gradient_batch_warmstart_faster_gaussian_t", type=float, default=1.0)

python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --affix "debug" --gpu 1 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method gpu_parallelized


python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 30 --train_attacker_use_n_step_schedule --affix "debug" --gpu 1 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian


python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 30 --train_attacker_use_n_step_schedule --affix "debug" --gpu 1 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method gpu_parallelized --gradient_batch_warmstart_faster_sampling_method gaussian


python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 30 --train_attacker_use_n_step_schedule --affix "debug" --gpu 1 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method uniform