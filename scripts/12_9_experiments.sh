# Old experiment:
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64 --gpu 1 --affix 64_64_40-2 --reg_weight 40 --random_seed 2 &> 64_64_40-2.out &

# Trying more random seeds on the successful experiment

#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 1 --affix 64_64_40-3 --reg_weight 40 --random_seed 3 &> 64_64_40-3.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 2 --affix 64_64_40-4 --reg_weight 40 --random_seed 4 &> 64_64_40-4.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 3 --affix 64_64_40-5 --reg_weight 40 --random_seed 5 &> 64_64_40-5.out &

#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 1 --affix 64_64_40-6 --reg_weight 40 --random_seed 6 &> 64_64_40-6.out &

# What happens with 64-64-64 if you don't average gradients?
nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64-64 --gpu 3 --affix 64_64_64_40-1 --reg_weight 40 --random_seed 1 &> 64_64_64_40-1.out &

nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64-64 --gpu 2 --affix 64_64_64_40-2 --reg_weight 40 --random_seed 2 &> 64_64_64_40-2.out &

nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 2500 --phi_nn_dimension 64-64-64 --gpu 1 --affix 64_64_64_45-1 --reg_weight 45 --random_seed 1 &> 64_64_64_45-1.out &


# 12/17
nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 1 --affix 64_64_40_no_adam --reg_weight 40 --random_seed 3 &> 64_64_60_no_adam.out &

nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 30 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 3000 --phi_nn_dimension 64-64 --gpu 2 --affix 64_64_40_no_adam_fast --reg_weight 40 --random_seed 3 &> 64_64_40_no_adam_fast.out &