
# Try theta dtheta again, quickly: sanity check "Is it bigger?"
nohup python -u flying_mpc.py --which_params theta dtheta --delta 0.5 --n_proc 12 &> mpc_theta_dtheta_fast.out &


#	parser.add_argument('--dt', type=float, default=0.05)
#	parser.add_argument('--delta', type=float, default=0.1, help="discretization of grid over slice")
#	parser.add_argument('--N_horizon', type=int, default=20)

nohup python -u flying_mpc.py --which_params theta dtheta --delta 0.5 --dt 0.01 --N_horizon 50 --n_proc 12 &> mpc_theta_dtheta_fast_smaller_dt.out &
nohup python -u flying_mpc.py --which_params theta dtheta --delta 0.5 --N_horizon 15 --n_proc 12 &> mpc_theta_dtheta_fast_horizon_15.out &
nohup python -u flying_mpc.py --which_params theta dtheta --delta 0.5 --N_horizon 10 --n_proc 12 &> mpc_theta_dtheta_fast_horizon_10.out &


nohup python -u flying_mpc.py --which_params theta dtheta --N_horizon 10 --n_proc 12 &> mpc_theta_dtheta_horizon_10.out &

# Trying new cost function
nohup python -u flying_mpc.py --which_params theta dtheta --n_proc 12 --affix theta_dtheta_new_cost &> mpc_theta_dtheta_new_cost.out &

# Try dphi-dgamma, dtheta-dbeta quickly: one sanity check is that they should be the same
nohup python -u flying_mpc.py --which_params dphi dgamma --delta 0.5 --n_proc 12 &> mpc_dphi_dgamma_fast.out &
nohup python -u flying_mpc.py --which_params dtheta dbeta --delta 0.5 --n_proc 12 &> mpc_dtheta_dbeta_fast.out &

# Same caption as above
#python flying_mpc.py --which_params gamma dgamma --delta 0.5
#python flying_mpc.py --which_params beta dbeta --delta 0.5

nohup python -u flying_mpc.py --which_params theta dtheta --delta 0.5 --n_proc 12 &> mpc_theta_dtheta_fast.out &
