# Reg weight 0
nohup python -u run_flying_pend_exps.py --save_fnm ckpt_160_vol_grid --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_0 --checkpoint_number_to_load 160 --which_experiments volume --volume_alg bfs_grid --bfs_axes_grid_size 0.23 0.23 0.70 3.33 3.33 3.33 0.23 0.23 3.33 3.33  &> reg_weight_0_vol_grid.out &

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_160_vol_sample --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_0 --checkpoint_number_to_load 160 --which_experiments volume --volume_alg sample --N_samp_volume 2500000 &> reg_weight_0_vol_sample.out &

# Reg weight 10

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_215_vol_grid --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_10 --checkpoint_number_to_load 215 --which_experiments volume --volume_alg bfs_grid --bfs_axes_grid_size 0.23 0.23 0.70 3.33 3.33 3.33 0.23 0.23 3.33 3.33  &> reg_weight_10_vol_grid.out &

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_215_vol_sample --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_10 --checkpoint_number_to_load 215 --which_experiments volume --volume_alg sample --N_samp_volume 2500000 &> reg_weight_10_vol_sample.out &

# Reg weight 50

#nohup python -u run_flying_pend_exps.py --save_fnm ckpt_290_vol_grid --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_50 --checkpoint_number_to_load 290 --which_experiments volume --volume_alg bfs_grid --bfs_axes_grid_size 0.23 0.23 0.70 3.33 3.33 3.33 0.23 0.23 3.33 3.33  &> reg_weight_50_vol_grid.out &

#nohup python -u run_flying_pend_exps.py --save_fnm ckpt_290_vol_sample --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_50 --checkpoint_number_to_load 290 --which_experiments volume --volume_alg sample --N_samp_volume 2500000 &> reg_weight_50_vol_sample.out &

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_180_vol_sample --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_50 --checkpoint_number_to_load 180 --which_experiments volume --volume_alg sample --N_samp_volume 2500000 &> reg_weight_50_vol_sample.out &

# Reg weight 100

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_490_vol_grid --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_200 --checkpoint_number_to_load 490 --which_experiments volume --volume_alg bfs_grid --bfs_axes_grid_size 0.23 0.23 0.70 3.33 3.33 3.33 0.23 0.23 3.33 3.33  &> reg_weight_200_vol_grid.out &

nohup python -u run_flying_pend_exps.py --save_fnm ckpt_490_vol_sample --which_cbf ours --exp_name_to_load flying_inv_pend_best_reg_weight_200 --checkpoint_number_to_load 490 --which_experiments volume --volume_alg sample --N_samp_volume 2500000 &> reg_weight_200_vol_sample.out &

