


# Without reg: can we successfully minimize the objective?

#nohup python -u baseline_run_cmaes.py --exp_name v1_n_feasible &> v1_n_feasible.out &
#
#nohup python -u baseline_run_cmaes.py --populate_num 500 --epoch 20 --init_sigma_ratio 0.5 --exp_name v2_n_feasible &> v2_n_feasible.out &
#
#nohup python -u baseline_run_cmaes.py --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --exp_name v3_n_feasible &> v3_n_feasible.out &



#nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name v1_avg_amount_infeasible &> v1_avg_amount_infeasible.out &
#
#nohup python -u baseline_run_cmaes.py --populate_num 500 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name v2_avg_amount_infeasible &> v2_avg_amount_infeasible &
#
#nohup python -u baseline_run_cmaes.py --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name v3_avg_amount_infeasible &> v3_avg_amount_infeasible.out &


nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name v1_max_amount_infeasible &> v1_max_amount_infeasible.out &

nohup python -u baseline_run_cmaes.py --populate_num 500 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name v2_max_amount_infeasible &> v2_max_amount_infeasible.out &

nohup python -u baseline_run_cmaes.py --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name v3_max_amount_infeasible &> v3_max_amount_infeasible.out &
