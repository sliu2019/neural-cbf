# 12_22: 64-64 gradient averaging, plus more attacks
# Should have fixed OOM

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 1 --affix 64_64_60pts_gradient_avging_seed_1 --reg_weight 40 --random_seed 1 &> 64_64_60pts_gradient_avging_seed_1.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 2 --affix 64_64_60pts_gradient_avging_seed_2 --reg_weight 40 --random_seed 2 &> 64_64_60pts_gradient_avging_seed_2.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 3 --affix 64_64_60pts_gradient_avging_seed_3 --reg_weight 40 --random_seed 3 &> 64_64_60pts_gradient_avging_seed_3.out &

nohup python main.py --train_attacker gradient_batch_warmstart --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --n_checkpoint_step 10 --trainer_stopping_condition n_steps --trainer_n_steps 3000 --phi_nn_dimension 64-64 --trainer_average_gradients --gpu 0 --affix 64_64_60pts_gradient_avging_seed_4 --reg_weight 40 --random_seed 4 &> 64_64_60pts_gradient_avging_seed_4.out &

