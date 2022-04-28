

# Full metrics on chosen "ours"

nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 &> compute_metric_flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0.out &

# Make tau smaller
nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_gaussian_t 0.1 --worst_boundary_gaussian_t 0.1 &> compute_metric_ours_tau_1e_1.out &

###################################
## Evaluate 4/26 CMAES runs as well
#nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1 --checkpoint_number_to_load 8 --which_experiments rollout --rollout_N_rollout 15 &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1.out &

#nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1 --checkpoint_number_to_load 8 &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1e_1 --checkpoint_number_to_load 8 &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1e_1.out &

# Rollouts only (just trying)

#nohup python -u run_flying_pend_exps.py --save_fnm default_rollouts_only --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1 --checkpoint_number_to_load 8 --which_experiments rollout &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm default_rollouts_only --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1e_1 --checkpoint_number_to_load 8 --which_experiments rollout &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1e_1.out &
