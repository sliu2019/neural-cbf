
############################################
### Debug mode

# Tau = 1.0
#python -u run_flying_pend_exps.py --save_fnm default --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments average_boundary worst_boundary volume --debug_mode
#
#python -u run_flying_pend_exps.py --save_fnm default_rollouts --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments rollout --debug_mode

# Tau = 0.1
#python -u run_flying_pend_exps.py --save_fnm tau_1e_1 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_gaussian_t 0.1 --worst_boundary_gaussian_t 0.1 --which_experiments average_boundary worst_boundary volume --debug_mode

#python -u run_flying_pend_exps.py --save_fnm tau_1e_1_rollouts --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_gaussian_t 0.1 --worst_boundary_gaussian_t 0.1 --which_experiments rollout --debug_mode

############################################
### Regular mode

# Tau = 1.0
python -u run_flying_pend_exps.py --save_fnm default --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments average_boundary worst_boundary volume

python -u run_flying_pend_exps.py --save_fnm default_rollouts --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --which_experiments rollout --which_analyses None

# Tau = 0.1
python -u run_flying_pend_exps.py --save_fnm tau_1e_1 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_gaussian_t 0.1 --worst_boundary_gaussian_t 0.1 --which_experiments average_boundary worst_boundary volume

python -u run_flying_pend_exps.py --save_fnm tau_1e_1_rollouts --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --boundary_gaussian_t 0.1 --worst_boundary_gaussian_t 0.1 --which_experiments rollout --which_analyses None