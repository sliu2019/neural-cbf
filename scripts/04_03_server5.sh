# server 5
## Euc, tanh-tanh-none
#nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "euc_seed_0" --gpu 0 &> euc_seed_0.out &
#nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "euc_seed_1" --gpu 0 &> euc_seed_1.out &
#nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "euc_seed_2" --gpu 1 &> euc_seed_2.out &
#
## Euc, tanh-tanh-softplus
#nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "euc_softplus_seed_0" --gpu 1 &> euc_softplus_seed_0.out &
#nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "euc_softplus_seed_1" --gpu 2 &> euc_softplus_seed_1.out &
#nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "euc_softplus_seed_2" --gpu 2 &> euc_softplus_seed_2.out &

# Euc, tanh-tanh-softplus, gradient averaging
nohup python main.py --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 0 --affix "euc_softplus_weighted_avg_seed_0" --gpu 3 &> euc_softplus_weighted_avg_seed_0.out &
nohup python main.py --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 1 --affix "euc_softplus_weighted_avg_seed_1" --gpu 3 &> euc_softplus_weighted_avg_seed_1.out &
#nohup python main.py --objective_option "weighted_average" --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "euc" --phi_include_xe --train_attacker_n_samples 100 --train_attacker_use_n_step_schedule --random_seed 2 --affix "euc_softplus_seed_2" --gpu 2 &> euc_softplus_seed_2.out &