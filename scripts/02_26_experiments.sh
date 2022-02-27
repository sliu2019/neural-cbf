
# Debugging reg term
python main.py --problem flying_inv_pend --h sum --phi_include_beta_deriv --no_softplus_on_obj --reg_weight 1.0 --phi_include_xe --train_attacker_n_samples 60 --train_attacker_max_n_steps 50 --train_attacker_projection_lr 1e-2 --trainer_stopping_condition n_steps --trainer_n_steps 4500 --affix debug --gpu 1