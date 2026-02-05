# server 4
# Spherical, tanh-tanh-none
nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 0 --affix "sphere_seed_0" --gpu 0 &> sphere_seed_0.out &
nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 1 --affix "sphere_seed_1" --gpu 0 &> sphere_seed_1.out &
nohup python main.py --phi_nnl "tanh-tanh-none" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 2 --affix "sphere_seed_2" --gpu 1 &> sphere_seed_2.out &

# Spherical, tanh-tanh-softplus
nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 0 --affix "sphere_softplus_seed_0" --gpu 1 &> sphere_softplus_seed_0.out &
nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 1 --affix "sphere_softplus_seed_1" --gpu 2 &> sphere_softplus_seed_1.out &
nohup python main.py --phi_nnl "tanh-tanh-softplus" --phi_nn_inputs "spherical" --phi_include_xe --critic_n_samples 100 --critic_use_n_step_schedule --random_seed 2 --affix "sphere_softplus_seed_2" --gpu 2 &> sphere_softplus_seed_2.out &
