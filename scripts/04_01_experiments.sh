
# server 5
# Euc, tanh-tanh-none
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "euc_seed_0" --gpu 0
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "euc_seed_1" --gpu 0
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "euc_seed_2" --gpu 1

# Euc, tanh-tanh-softplus
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "euc_softplus_seed_0" --gpu 1
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "euc_softplus_seed_1" --gpu 2
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "euc_softplus_seed_2" --gpu 2

# server 4
# Spherical, tanh-tanh-none
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "sphere_seed_0" --gpu 0
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "sphere_seed_1" --gpu 0
python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "sphere_seed_2" --gpu 1

# Spherical, tanh-tanh-softplus
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "sphere_softplus_seed_0" --gpu 1
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "sphere_softplus_seed_1" --gpu 2
python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "sphere_softplus_seed_2" --gpu 2



## reg
## Tune weight before running
#--reg_weight
#--objective_option "weighted_average"
#
## trainer
#--train_attacker_n_samples 100 --train_attacker_use_n_step_schedule
#
##
#--random_seed 1
#--affix hey
#--trainer_n_steps
#
## misc
#n_checkpoint_step 5
#n_test_loss_step 50

#problem          : flying_inv_pend
#h                : sum
#phi_nn_dimension : 32-32
#phi_nnl          : tanh-tanh-softplus
#phi_ci_init_range : 0.01
#phi_include_xe   : True
#phi_nn_inputs    : euc
#physical_difficulty : easy

#pend_length      : 3.0
#box_ang_vel_limit : 20.0

#reg_weight       : 10.0
#reg_sample_distance : 0.1
#reg_sampler      : random
#reg_n_samples    : 250

#objective_option : weighted_average
#train_attacker   : gradient_batch_warmstart
#gradient_batch_warmstart2_proj_tactic : None
#train_attacker_n_samples : 10
#train_attacker_stopping_condition : n_steps
#train_attacker_max_n_steps : 2
#train_attacker_projection_tolerance : 0.1
#train_attacker_projection_lr : 0.01
#train_attacker_projection_time_limit : 3.0
#train_attacker_lr : 0.001
#train_attacker_use_n_step_schedule : False
#trainer_stopping_condition : n_steps
#trainer_early_stopping_patience : 100
#trainer_n_steps  : 4000
#trainer_lr       : 0.001
#random_seed      : 1
#affix            : debug
#log_root         : log
#model_root       : checkpoint
#n_checkpoint_step : 10
#n_test_loss_step : 10
#gpu              : 0
#log_folder       : log/flying_inv_pend_debug
#model_folder     : checkpoint/flying_inv_pend_debug
#Using GPU device: cuda:0