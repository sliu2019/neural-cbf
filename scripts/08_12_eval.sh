# Redoing rollouts, since limits were not applied correctly

# Ours, LQR, q = 0.1
nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1e_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 0.1 --run_length long &> lqr_ours_q_1e_1.out &

# Ours, LQR, q = 1.0
nohup python -u run_flying_pend_exps.py --save_fnm LQR_q_1_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 1.0 --run_length long &> lqr_ours_q_1.out &

# Ours, un-actuated q = 1.0
nohup python -u run_flying_pend_exps.py --save_fnm unactuated_long --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref unactuated --rollout_T_max 2.5 --run_length long &> unactuated_long.out &


#python -u run_flying_pend_exps.py --save_fnm debug_LQR --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --which_experiments rollout --rollout_u_ref LQR --rollout_T_max 2.5 --rollout_LQR_q 0.1 --run_length short