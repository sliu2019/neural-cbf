
#python -u run_flying_pend_exps.py --save_fnm long_length_ckpt_250 --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --run_length long

#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init_new_ub_reg_weight_1 --checkpoint_number_to_load 2 --run_length long &> eval_cmaes_UB_long.out &

# Debug the LQR controller
#python -u run_flying_pend_exps.py --save_fnm test_LQR --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_N_rollout 5 --rollout_u_ref LQR --rollout_T_max 2.5
#
## Baseline
#python -u run_flying_pend_exps.py --save_fnm test_LQR --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init --checkpoint_number_to_load 10 --which_experiments rollout --rollout_N_rollout 5 --rollout_u_ref LQR --rollout_T_max 2.5


#	parser.add_argument('--rollout_u_ref', type=str, choices=["unactuated", "LQR", "MPC"], default="unactuated")
#	parser.add_argument('--rollout_LRQ_q', type=float, default=1.0)
#	parser.add_argument('--rollout_LRQ_r', type=float, default=1.0)
# parser.add_argument('--rollout_T_max', type=float, default=1.0)

# Run LQR with different q, r parameters

# q = 1.0, r = 1.0
nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1_r_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long &> lqr_ours_q_1_r_1.out &

#nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1_r_1_long --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init --checkpoint_number_to_load 10 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long &> lqr_cmaes_q_1_r_1.out &

# q = 2.0, r = 1.0
nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_2_r_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long --rollout_LRQ_q 2.0 &> lqr_ours_q_2_r_1.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_2_r_1_long --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init --checkpoint_number_to_load 10 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long --rollout_LRQ_q 2.0 &> lqr_cmaes_q_2_r_1.out &


# q = 0.5, r = 1.0
nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_point5_r_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long --rollout_LRQ_q 0.5 &> lqr_ours_q_point5_r_1.out &

#nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_point5_r_1_long --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init --checkpoint_number_to_load 10 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --run_length long --rollout_LRQ_q 0.5 &> lqr_cmaes_q_point5_r_1.out &


# TODO: Volume eval

nohup python -u run_flying_pend_exps.py --save_fnm it_125_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0_reg_0 --checkpoint_number_to_load 125 --which_experiments volume --run_length long &> volume_no_reg_it_125.out &

nohup python -u run_flying_pend_exps.py --save_fnm it_425_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0_reg_0 --checkpoint_number_to_load 425 --which_experiments volume --run_length long &> volume_no_reg_it_425.out &


#### 06/19
python -u run_flying_pend_exps.py --save_fnm tau_tiny --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments average_boundary --boundary_gaussian_t 1e-5 --boundary_n_samples 1000


nohup python -u run_flying_pend_exps.py --save_fnm tau_super_tiny --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments average_boundary --boundary_gaussian_t 1e-7 --boundary_n_samples 1000 &> tau_super_tiny.out &


### Tuesday, June 21st


python -u run_flying_pend_exps.py --save_fnm debug_LQR --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 10 --rollout_dt 0.1 --rollout_LQR_q 2.0 --rollout_N_rollout 1

