
# Server 5
#nohup python flying_mpc.py --which_params dgamma dphi --n_proc 12 &> dgamma_dphi.out &
#
#nohup python flying_mpc.py --which_params dbeta dtheta --n_proc 12 &> dbeta_dtheta.out &
#
#nohup python flying_mpc.py --which_params phi dphi --n_proc 12 &> phi_dphi.out &


# Server 4
nohup python flying_mpc.py --which_params theta dtheta --n_proc 12 &> theta_dtheta.out &

nohup python flying_mpc.py --which_params beta dbeta --n_proc 12 --N_horizon 50 &> beta_dbeta.out &

nohup python flying_mpc.py --which_params gamma dgamma --n_proc 12 --N_horizon 50 &> gamma_dgamma.out &


