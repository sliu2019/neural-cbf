# S nonempty

nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0 --affix exp1a --n_checkpoint_step 30 &> exp1a.out &
nohup python -u main.py --problem cartpole_reduced --physical_difficulty easy --objective_volume_weight 0.025 --affix exp1b --n_checkpoint_step 30 &> exp1b.out &
nohup python -u main.py --problem cartpole_reduced --physical_difficulty hard --objective_volume_weight 0 --affix exp1c --n_checkpoint_step 30 &> exp1c.out &

#- Exp 1a: 2 vars, weight = 0, easy physical parameters
#- Exp 1b: 2 vars, weight = 0.025, easy physical parameters
#- Exp 1c: 2 vars, weight = 0, hard physical parameters

# python -u main.py --problem cartpole_reduced --affix debug