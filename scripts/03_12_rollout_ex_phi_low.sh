
# Want these to execute sequentially, since they each try to use all of the CPU cores
# To run this file: nohup fname &> file1.out &

#python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --N_rollout 250
#
#python flying_rollout_experiment.py --which_cbf low --low_c2 1.0 --N_rollout 250
#
#python flying_rollout_experiment.py --which_cbf low --low_c2 0.01 --N_rollout 250


# Note: c3 = 10.0 too large, cannot find x0
#python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --low_c3 10.0 --N_rollout 250


#python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --low_c3 1.0 --N_rollout 250
python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --low_c3 2.0 --N_rollout 250
python flying_rollout_experiment.py --which_cbf low --low_c2 0.1 --low_c3 3.0 --N_rollout 250