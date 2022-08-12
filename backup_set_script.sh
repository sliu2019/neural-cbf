nohup python -u backup_set_methods.py --q 0.05 &> r1.out &

nohup python -u backup_set_methods.py --param1 gamma --param2 dgamma &> r2.out &

# Didn't work
#nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --delta 0.1 &> r3.out &

nohup python -u backup_set_methods.py --param1 dtheta --param2 dbeta --delta 0.01 --T 30 &> r3.out &

nohup python -u backup_set_methods.py --q 0.25 --param1 dtheta --param2 dbeta --delta 0.01 --T 30 &> r4.out &