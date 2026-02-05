nohup python main.py --affix 64_64_gradient_avging_seed_2 --random_seed 2 --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 0 &> 64_64_gradient_avging_seed_2.out &

nohup python main.py --affix 64_64_gradient_avging_seed_3 --random_seed 3 --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 1 &> 64_64_gradient_avging_seed_3.out &

nohup python main.py --affix 64_64_gradient_avging_seed_4 --random_seed 4 --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 2 &> 64_64_gradient_avging_seed_4.out &

nohup python main.py --affix 64_64_gradient_avging_seed_5 --random_seed 5 --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64 --learner_average_gradients --gpu 3 &> 64_64_gradient_avging_seed_5.out &
