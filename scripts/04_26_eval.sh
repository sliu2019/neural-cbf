
# First time evaluating

# Lot of samples
#nohup python -u run_flying_pend_exps.py --save_fnm many_samples --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 &> compute_metric_flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0.out &

# Medium amount of samples; will run quicker so we can get an idea immediately
#nohup python -u run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_n_samples 100 --worst_boundary_n_samples 100 --rollout_N_rollout 100 &> faster_compute_metric_flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0.out &

# Debug rollout's failure to terminate
#python -u run_flying_pend_exps.py --save_fnm debug_rollout_inf_loop --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments rollout --rollout_N_rollout 500 --rollout_T_max 1.0

python -u run_flying_pend_exps.py --save_fnm throwaway_actually --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments rollout --rollout_N_rollout 500 --rollout_T_max 1.0

python -u run_flying_pend_exps.py --save_fnm throwaway_actually --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments rollout --rollout_N_rollout 2 --rollout_T_max 1.0

# Evaluate 4/26 CMAES runs as well
nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1 --checkpoint_number_to_load 8 &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1.out &

nohup python -u run_flying_pend_exps.py --save_fnm default --which_cbf low-CMAES --exp_name_to_load flying_pend_avg_amount_infeasible_reg_weight_1e_1 --checkpoint_number_to_load 8 &> compute_metric_flying_pend_avg_amount_infeasible_reg_weight_1e_1.out &