nohup python mpc_inverted_pendulum.py --dt 0.02 --N_horizon 65 &> mpc_dt_0.02_horizon_65.out &
nohup python mpc_inverted_pendulum.py --dt 0.02 --N_horizon 50 --delta 0.05 &> mpc_dt_0.02_horizon_50_delta_0.05.out &
