


nohup python -u baseline_run_cmaes.py --epoch 25 --FlyingPendEvaluator_reg_weight 0.0 --FlyingPendEvaluator_n_samples 1000 --exp_name mostly_default_no_reg &> mostly_default_no_reg.out &
nohup python -u baseline_run_cmaes.py --epoch 25 --populate_num 100 --FlyingPendEvaluator_reg_weight 0.0 --FlyingPendEvaluator_n_samples 1000 --exp_name pop_num_100_no_reg &> pop_num_100_no_reg.out &



python -u baseline_run_cmaes.py --epoch 25 --populate_num 100 --FlyingPendEvaluator_reg_weight 0.0 --exp_name debug