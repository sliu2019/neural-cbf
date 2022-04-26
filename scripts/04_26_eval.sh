
# First time evaluating

# Lot of samples
nohup python -u run_flying_pend_exps.py --save_fnm many_samples --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 &> compute_metric_flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0.out &

# Medium amount of samples; will run quicker so we can get an idea immediately
#nohup python -u run_flying_pend_exps.py --save_fnm debug --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_n_samples 100 --worst_boundary_n_samples 100 --rollout_N_rollout 100 &> faster_compute_metric_flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0.out &