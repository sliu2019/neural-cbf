
# Objective: avg_amount_infeasible
nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1 &> avg_amount_infeasible_reg_weight_1.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1e_1 &> avg_amount_infeasible_reg_weight_1e_1.out &

# Objective: max_amount_infeasible
nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name max_amount_infeasible_reg_weight_1 &> max_amount_infeasible_reg_weight_1.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name max_amount_infeasible_reg_weight_1e_1 &> max_amount_infeasible_reg_weight_1e_1.out &

