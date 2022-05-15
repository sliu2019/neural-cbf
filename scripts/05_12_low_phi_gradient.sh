
# Debug low-phi gradient

#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 0.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf" --gpu 0 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf.out &

#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 1.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_1" --gpu 0 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf_reg_weight_1.out &
#
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 10.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_10" --gpu 1 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf_reg_wiehgt_10.out &
#
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 100.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_100" --gpu 2 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf_reg_weight_100.out &


#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 500 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 1 --affix "ESG_reg_speedup_better_attacks_seed_1" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_1.out &


#####################################################################

# Friday, May 13 runs

# Figure out proper reg weight
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 1.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_1" --gpu 0 &> gradient_train_low_cbf_reg_weight_1.out &

# Try 10 seeds

#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 1.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_debug" --gpu 0 &> gradient_train_low_cbf_reg_weight_1.out &
#
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 10.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_10" --gpu 1 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf_reg_wiehgt_10.out &
#
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 100.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_100" --gpu 2 --n_checkpoint_step 1 --n_test_loss_step 1 &> gradient_train_low_cbf_reg_weight_100.out &
#
#
#nohup python main.py --phi_design "low" --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 1.0 --objective_option "weighted_average" --trainer_lr 1e-2 --train_attacker_n_samples 50 --train_attacker_use_n_step_schedule --train_attacker_max_n_steps 20 --train_attacker_p_reuse 0.5 --train_attacker_lr 1e-4 --train_attacker_projection_lr 1e-3 --train_attacker_projection_time_limit 5.0 --train_attacker gradient_batch_warmstart_faster --gradient_batch_warmstart_faster_speedup_method sequential --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "low_cbf_reg_weight_1" --gpu 0 &> gradient_train_low_cbf_reg_weight_1.out &






