# S nonempty

nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0 --affix exp1a --n_checkpoint_step 30 --gpu 0 &> exp1a.out &
nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0.025 --affix exp1b --n_checkpoint_step 30 --gpu 1 &> exp1b.out &
#nohup python -u main.py --problem cartpole_reduced --physical_difficulty hard --objective_volume_weight 0 --affix exp1c --n_checkpoint_step 30 --gpu 2 &> exp1c.out &

#- Exp 1a: 2 vars, weight = 0, easy physical parameters
#- Exp 1b: 2 vars, weight = 0.025, easy physical parameters
#- Exp 1c: 2 vars, weight = 0, hard physical parameters

# python -u main.py --problem cartpole_reduced --affix debug


python -u main.py --affix exp1a --n_checkpoint_step 3 --critic_projection_stop_threshold 1e-1 --test_critic_projection_stop_threshold 1e-1 --critic_projection_lr 1.0 --test_critic_projection_lr 1.0 --learner_early_stopping_patience 10

python -u main.py --affix exp1a --n_checkpoint_step 1 --learner_early_stopping_patience 10 --critic_projection_stop_threshold 1e-2 --test_critic_projection_stop_threshold 1e-2