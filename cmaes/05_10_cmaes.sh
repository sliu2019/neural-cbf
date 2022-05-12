
# New init
#nohup python -u baseline_run_cmaes.py --init_params 0.1 0.0 1.0 --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name new_init &> new_init.out &
#
#nohup python -u baseline_run_cmaes.py --init_params 0.1 0.0 1.0 --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.3 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name new_init_sigma_ratio_3e_1 &> new_init_sigma_ratio_3e_1.out &
#
#nohup python -u baseline_run_cmaes.py --init_params 0.1 0.0 1.0 --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name new_init_reg_weight_1e_1 &> new_init_reg_weight_1e_1.out &

# New init, new ub
nohup python -u baseline_run_cmaes.py --init_params 0.1 0.0 1.0 --upper_bound 5.0 5.0 5.0 --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 10 --init_sigma_ratio 0.3 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name new_init_new_ub_reg_weight_1 &> new_init_new_ub_reg_weight_1.out &

nohup python -u baseline_run_cmaes.py --init_params 0.1 0.0 1.0 --upper_bound 5.0 5.0 5.0 --FlyingPendEvaluator_reg_weight 1e-1 --populate_num 500 --elite_ratio 1e-2 --epoch 10 --init_sigma_ratio 0.3 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name new_init_new_ub_reg_weight_1e_1 &> new_init_new_ub_reg_weight_1e_1.out &

