#nohup python main.py --affix 64_64_gradient_avging_1 --reg_weight 1 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 1000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 3 &> 64_64_gradient_avging_1.out &
#
#nohup python main.py --affix 64_64_gradient_avging_10 --reg_weight 10 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 1000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 0 &> 64_64_gradient_avging_10.out &
#
#nohup python main.py --affix 64_64_gradient_avging_50 --reg_weight 50 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 1000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 1 &> 64_64_gradient_avging_50.out &
#
#nohup python main.py --affix 64_64_gradient_avging_100 --reg_weight 100 --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 1000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 2 &> 64_64_gradient_avging_100.out &


# Evening
#nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 0 --affix 64_64_gradient_avging_30-1 --reg_weight 30 --random_seed 1 &> 64_64_gradient_avging_30-1.out &
#
#nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 1 --affix 64_64_gradient_avging_30-2 --reg_weight 30 --random_seed 2 &> 64_64_gradient_avging_30-2.out &
#
#
#
#nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 2 --affix 64_64_gradient_avging_40-1 --reg_weight 40 --random_seed 1 &> 64_64_gradient_avging_40-1.out &
#
#nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 3 --affix 64_64_gradient_avging_40-2 --reg_weight 40 --random_seed 2 &> 64_64_gradient_avging_40-2.out &
#


nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --gpu 0 --affix 64_64_40-1 --reg_weight 40 --random_seed 1 &> 64_64_40-1.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64 --gpu 1 --affix 64_64_40-2 --reg_weight 40 --random_seed 2 &> 64_64_40-2.out &


nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64-64 --trainer_average_gradients --gpu 2 --affix 64_64_64_gradient_avging_40-1 --reg_weight 40 --random_seed 1 &> 64_64_64_gradient_avging_40-1.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 30 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 2500 --phi_nn_dimension 64-64-64 --trainer_average_gradients --gpu 3 --affix 64_64_64_gradient_avging_40-2 --reg_weight 40 --random_seed 2 &> 64_64_64_gradient_avging_40-2.out &
