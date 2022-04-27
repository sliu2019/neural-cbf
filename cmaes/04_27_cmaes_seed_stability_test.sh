
### Seeing if solution (final param set) is stable to a change of seeds

# Gonna try 4 random seeds each for 2 experiments (seeds 1,2,3; we already have 0)

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1 --random_seed 1 &> avg_amount_infeasible_reg_weight_1_seed_1.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1 --random_seed 2 &> avg_amount_infeasible_reg_weight_1_seed_2.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1 --random_seed 3 &> avg_amount_infeasible_reg_weight_1_seed_3.out &

#################################################################################

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1e_1 --random_seed 1 &> avg_amount_infeasible_reg_weight_1e_1_seed_1.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1e_1 --random_seed 2 &> avg_amount_infeasible_reg_weight_1e_1_seed_2.out &

nohup python -u baseline_run_cmaes.py --FlyingPendEvaluator_reg_weight 0.1 --populate_num 500 --elite_ratio 1e-2 --epoch 20 --init_sigma_ratio 0.5 --FlyingPendEvaluator_objective_type avg_amount_infeasible --exp_name avg_amount_infeasible_reg_weight_1e_1 --random_seed 3 &> avg_amount_infeasible_reg_weight_1e_1_seed_3.out &