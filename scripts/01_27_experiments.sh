# Starting flying inv pend experiments

python main.py --problem flying_inv_pend --phi_include_xe --reg_weight 0.0 --trainer_stopping_condition n_steps --trainer_n_steps 1500 --affix debug

# Other params you might want to tune:
# --reg_sample_distance 0.1


# Later:
# k0, ci init
# reg_weight: 40

# For experiments:
# Fix random seed, gpu number
