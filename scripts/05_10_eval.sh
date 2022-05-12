
# Validated 3 changes to run_flying_pend_exps.py

#python -u run_flying_pend_exps.py --save_fnm medium_length --which_cbf low-heuristic --low_cbf_params 0.1 0.0 1.0--run_length short

#python -u run_flying_pend_exps.py --save_fnm medium_length --which_cbf low-heuristic --low_cbf_params 0.1 0.0 1.0 --run_length medium

# Low heuristic
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf low-heuristic --low_cbf_params 0.1 0.0 1.0 --run_length long &> eval_heuristic_long.out &

# Low CMAES, new init
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init --checkpoint_number_to_load 10 --run_length long &> eval_cmaes_long.out &

# Low CMAES, new init
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init_reg_weight_1e_1 --checkpoint_number_to_load 10 --run_length long &> eval_cmaes_smaller_reg_long.out &

# Low CMAES, upper bound
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf low-CMAES --exp_name_to_load flying_pend_new_init_new_ub_reg_weight_1 --checkpoint_number_to_load 2 --run_length long &> eval_cmaes_UB_long.out &

# Ours, long experiment (more samples)
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 1020 --run_length long &> eval_ours_long.out &

# Ours, a few more epochs and seeds

#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_0 --checkpoint_number_to_load 250 --run_length long &> eval_ours_seed0_long.out &

#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_1 --checkpoint_number_to_load 375 --run_length long &> eval_ours_seed1_long.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_2 --checkpoint_number_to_load 625 --run_length long &> eval_ours_seed2_long.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_3 --checkpoint_number_to_load 250 --run_length long &> eval_ours_seed3_long.out &
#
#nohup python -u run_flying_pend_exps.py --save_fnm long_length --which_cbf ours --exp_name_to_load flying_inv_pend_ESG_reg_speedup_better_attacks_seed_4 --checkpoint_number_to_load 175 --run_length long &> eval_ours_seed4_long.out &