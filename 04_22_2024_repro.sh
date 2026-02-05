## Reproducing as a starting point for nCBF-drone project
## Want to reproduce results from rebuttal period

# The original
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --feas_loss_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_critic_n_samples 500 --train_critic_use_n_step_schedule --train_critic_max_n_steps 20 --train_critic_p_reuse 0.5 --train_critic boundary --boundary_speedup_method sequential --boundary_sampling_method gaussian --random_seed 0 --affix "ESG_reg_speedup_better_attacks_seed_0" --gpu 0 &> ESG_reg_speedup_better_attacks_seed_0.out &

# Reproduce: critic bs = 50/100 takes 2 hours to complete
 nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --feas_loss_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_critic_n_samples 50 --train_critic_use_n_step_schedule --train_critic_max_n_steps 20 --train_critic_p_reuse 0.5 --train_critic boundary --boundary_speedup_method sequential --boundary_sampling_method gaussian --random_seed 0 --affix "best_critic_bs_50_repro" --gpu 1 &> best_critic_bs_50_repro.out &

# Reproduce: reg_weight = 200 leads to a safe set of a certain volume
#nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --feas_loss_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_critic_n_samples 100 --train_critic_use_n_step_schedule --train_critic_max_n_steps 20 --train_critic_p_reuse 0.5 --train_critic boundary --boundary_speedup_method sequential --boundary_sampling_method gaussian --random_seed 0 --learner_n_steps 1500 --affix "best_reg_weight_200" --gpu 1 &> best_reg_weight_200.out &

# New experiment: combine lessons from the rebuttal period
nohup python main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 200.0 --feas_loss_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_critic_n_samples 100 --train_critic_use_n_step_schedule --train_critic_max_n_steps 20 --train_critic_p_reuse 0.0 --train_critic boundary --boundary_speedup_method sequential --boundary_sampling_method gaussian --random_seed 0 --learner_n_steps 1500 --affix "combined_rebuttal_takeaways" --gpu 2 &> combined_rebuttal_takeaways.out &


# 02-25-26 new commands 
# 4 PM EST
nohup python -u main.py --reg_transform "sigmoid" --reg_sampler "random_inside" --reg_weight 150.0 --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --critic_n_samples 500 --critic_use_n_step_schedule --critic_max_n_steps 20 --critic_p_reuse 0.5 --gradient_batch_warmstart_faster_sampling_method gaussian --random_seed 0 --affix "repro_after_delete" --gpu 0 &> repro_after_delete.out &