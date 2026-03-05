#mnist
# one round time

# Niss
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --k 0.0 --account --account1

# Chain-PPFL
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1

# Secagg
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1 --d 0.1

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1 --d 0.2

# dp-nsa 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1 --d 0.1

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --account1 --d 0.2