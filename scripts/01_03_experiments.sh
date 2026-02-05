# critic_n_samples: 30, 45, 60? 60, don't take chances
# ci: small range, to force mode 2 (k1 = 0)
# --phi_include_xe
# --reg_xe

# Round 1
##########################
## Exp 1: no x_e, volume reg only
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 0 --affix debugpinch1 --reg_weight 40 --random_seed 1 --phi_k0_init_max 10 &> debugpinch1.out &
#
## Exp 2: no x_e, volume and x_e reg
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch2a --reg_weight 40 --random_seed 1 --reg_xe 45 --phi_k0_init_max 10 &> debugpinch2a.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch2b --reg_weight 40 --random_seed 1 --reg_xe 55 --phi_k0_init_max 10 &> debugpinch2b.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 0 --affix debugpinch2c --reg_weight 40 --random_seed 1 --reg_xe 65 --phi_k0_init_max 10 &> debugpinch2c.out &
#
## Exp 3: include x_e
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 3 --affix debugpinch3 --reg_weight 40 --random_seed 1 --phi_include_xe &> debugpinch3.out &

##########################
# Round 2
# Exp 1: no x_e, volume reg only
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 0 --affix debugpinch1_softplus_s1 --reg_weight 40 --random_seed 1 --phi_k0_init_max 10 &> debugpinch1_softplus_s1.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 0 --affix debugpinch1_softplus_s2 --reg_weight 40 --random_seed 2 --phi_k0_init_max 10 &> debugpinch1_softplus_s2.out &
#
## Note: had to set k0 larger to make the invariant nonempty for this random seed. Gives k0 ~= 1.5
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch1_softplus_s3 --reg_weight 40 --random_seed 3 --phi_k0_init_max 15 &> debugpinch1_softplus_s3.out &
#
## Exp 3: include x_e
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch3_softplus_s1 --reg_weight 40 --random_seed 1 --phi_include_xe &> debugpinch3_softplus_s1.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 3 --affix debugpinch3_softplus_s2 --reg_weight 40 --random_seed 2 --phi_include_xe &> debugpinch3_softplus_s2.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 3 --affix debugpinch3_softplus_s3 --reg_weight 40 --random_seed 3 --phi_include_xe &> debugpinch3_softplus_s3.out &

##########################
# Round 3
# Exp 1: no x_e, volume reg only
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 2 --affix debugpinch1_resoftplus_s1 --reg_weight 40 --random_seed 1 --phi_k0_init_max 10 &> debugpinch1_resoftplus_s1.out &
#
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 2 --affix debugpinch1_resoftplus_s2 --reg_weight 40 --random_seed 2 --phi_k0_init_max 10 &> debugpinch1_resoftplus_s2.out &
#
## Note: had to set k0 larger to make the invariant nonempty for this random seed. Gives k0 ~= 1.5
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 3 --affix debugpinch1_resoftplus_s3 --reg_weight 40 --random_seed 3 --phi_k0_init_max 15 &> debugpinch1_resoftplus_s3.out &

# Exp 3: include x_e
nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 2 --affix debugpinch3_resoftplus_s1 --reg_weight 40 --random_seed 1 --phi_include_xe &> debugpinch3_resoftplus_s1.out &

nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch3_resoftplus_s2 --reg_weight 40 --random_seed 2 --phi_include_xe &> debugpinch3_resoftplus_s2.out &

# Large k0 spread
#nohup python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 1 --affix debugpinch3_resoftplus_s3 --reg_weight 40 --random_seed 3 --phi_include_xe --phi_k0_init_max 15 &> debugpinch3_resoftplus_s3.out &

# Debug slow + memory allocation
#python main.py --critic gradient_batch_warmstart --critic_n_samples 60 --critic_max_n_steps 50 --n_checkpoint_step 10 --learner_stopping_condition n_steps --learner_n_steps 1500 --phi_nn_dimension 64-64 --gpu 3 --affix debugpinch3_resoftplus_s2 --reg_weight 40 --random_seed 2 --phi_include_xe --phi_k0_init_max 1
