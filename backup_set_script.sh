nohup python -u backup_set_methods.py --q 0.05 &> theta_dtheta_smaller_q.out &

nohup python -u backup_set_methods.py --param1 gamma --param2 dgamma &> gamma_dgamma.out &

# Didn't work
#nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --delta 0.1 &> r3.out &

nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --delta 0.1 &> dtheta_dbeta_fast.out &

nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --q 0.5 --delta 0.1 &> dtheta_dbeta_fast_larger_q.out &

nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --q 0.05 --delta 0.1 &> dtheta_dbeta_fast_smaller_q.out &

nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --delta 0.1 --T 35 &> dtheta_dbeta_fast_longer_T.out &

#nohup python -u backup_set_methods.py --q 0.25 --param1 dtheta --param2 dbeta --delta 0.01 --T 30 &> r4.out &