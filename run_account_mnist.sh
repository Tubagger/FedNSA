# 25clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 50clients
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 75clients
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 100clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 