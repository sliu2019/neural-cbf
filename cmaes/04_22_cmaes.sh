# Repeating one successful experiment from last night to see if the result is stable across random seeds
#nohup python -u baseline_run_cmaes.py --exp_name v1_n_feasible_seed_0 &> v1_n_feasible_seed_0.out &
#nohup python -u baseline_run_cmaes.py --exp_name v1_n_feasible_seed_1 --random_seed 1 &> v1_n_feasible_seed_1.out &
#nohup python -u baseline_run_cmaes.py --exp_name v1_n_feasible_seed_2 --random_seed 2 &> v1_n_feasible_seed_2.out &
#nohup python -u baseline_run_cmaes.py --exp_name v1_n_feasible_seed_3 --random_seed 3 &> v1_n_feasible_seed_3.out &

# Reg weight tuning, for all 3 objectives
# Default in Tianhao's code is 1.0
# Running the only version that converged, but took five-ever to converge

# SaturationRisk: n_feasible
nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1e-1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --exp_name n_feasible_reg_weight_1e_1 &> n_feasible_reg_weight_1e_1.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 5e-2 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --exp_name n_feasible_reg_weight_5e_2 &> n_feasible_reg_weight_5e_2.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1e-2 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --exp_name n_feasible_reg_weight_1e_2 &> n_feasible_reg_weight_1e_2.out &

# SaturationRisk: avg_amount_infeasible
nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 10 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_10 &> avg_amount_infeasible_reg_weight_10.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 50 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_50 &> avg_amount_infeasible_reg_weight_50.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 100 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_100 &> avg_amount_infeasible_reg_weight_100.out &

# SaturationRisk: max_amount_infeasible
nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 15 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name max_amount_infeasible_reg_weight_15 &> max_amount_infeasible_reg_weight_15.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 75 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name max_amount_infeasible_reg_weight_75 &> max_amount_infeasible_reg_weight_75.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 150 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type max_amount_infeasible --exp_name max_amount_infeasible_reg_weight_150 &> max_amount_infeasible_reg_weight_150.out &

