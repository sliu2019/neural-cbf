#		parser.add_argument('--dynamics_noise_spread', type=float, default=0.0, help='set std dev of zero-mean, Gaussian noise')
#		parser.add_argument('--mismatched_model_parameter', type=str)
#		parser.add_argument('--mismatched_model_parameter_true_value', type=float)

# Server 5
# Repro test
#nohup python -u run_flying_pend_exps.py --save_fnm repro_test_eval --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref unactuated --rollout_T_max 2.5 --run_length long &> repro_test_eval.out &
#
## Robustness under LQR and noise
#nohup python -u run_flying_pend_exps.py --save_fnm noise_LQR_0_1 --dynamics_noise_spread 0.1 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> noise_LQR_0_1.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm noise_LQR_0_5 --dynamics_noise_spread 0.5 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> noise_LQR_0_5.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm noise_LQR_1 --dynamics_noise_spread 1.0 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> noise_LQR_1.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm noise_LQR_2 --dynamics_noise_spread 2.0 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> noise_LQR_2.out &
# Server 4
# Debug BFS volume algorithm
nohup python -u run_flying_pend_exps.py --save_fnm debug_bfs --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments volume --rollout_u_ref unactuated --rollout_T_max 2.5 --run_length long --volume_alg bfs_grid --bfs_axes_grid_size 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 &> debug_bfs.out &

nohup python -u run_flying_pend_exps.py --save_fnm debug_bfs_2 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments volume --volume_alg bfs_grid --bfs_axes_grid_size 1.0 1.0 1.0 5.0 5.0 5.0 1.0 1.0 5.0 5.0 &> debug_bfs_2.out &

# Repro test
nohup python -u run_flying_pend_exps.py --save_fnm repro_test_eval --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref unactuated --rollout_T_max 2.5 --run_length long &> repro_test_eval.out &

# Debug
#python -u run_flying_pend_exps.py --mismatched_model_parameter J_x J_y J_z --mismatched_model_parameter_true_value 0.01 0.01 0.018 --save_fnm mismatch_LQR_2 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --rollout_N_rollout 10

# Robustness under LQR and model mismatch
nohup python -u run_flying_pend_exps.py --mismatched_model_parameter J_x J_y J_z --mismatched_model_parameter_true_value 0.01 0.01 0.018 --save_fnm mismatch_LQR_2 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> mismatch_LQR_2.out &

nohup python -u run_flying_pend_exps.py --mismatched_model_parameter J_x J_y J_z --mismatched_model_parameter_true_value 0.025 0.025 0.045 --save_fnm mismatch_LQR_5 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> mismatch_LQR_5.out &

nohup python -u run_flying_pend_exps.py --mismatched_model_parameter J_x J_y J_z --mismatched_model_parameter_true_value 0.05 0.05 0.09 --save_fnm mismatch_LQR_10 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> mismatch_LQR_10.out &

#nohup python -u run_flying_pend_exps.py --mismatched_model_parameter J_x J_y J_z --mismatched_model_parameter_true_value 0.10 0.10 0.18 --save_fnm mismatch_LQR_20 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> mismatch_LQR_20.out &

# Debug
#python -u run_flying_pend_exps.py --save_fnm debug_eval --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref unactuated --rollout_T_max 2.5 --rollout_N_rollout 5 --dynamics_noise_spread 0.1
#
## Ours, LQR, q = 0.1
#nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1e_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 0.1 --run_length long &> lqr_ours_q_1e_1.out &
#
## Ours, LQR, q = 1.0
#nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> lqr_ours_q_1.out &
#
## Ours, un-actuated
#nohup python -u run_flying_pend_exps.py --save_fnm unactuated_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref unactuated --rollout_T_max 2.5 --run_length long &> unactuated_long.out &