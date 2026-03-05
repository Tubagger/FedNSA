# one round time
# mnist
# Niss
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --k 0.0 --account2

# Chain-PPFL
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account2

# Secagg
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account2

# dp-nsa 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account2

#cifar10
# Niss
python3  main.py --gpu 0 --dataset cifar_10 --model  squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.001 --iid  --local_bs 32 --local_ep 5 --k 0.0 --account2

# Chain-PPFL
python3  main.py --gpu 0 --dataset cifar_10 --model  squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.001 --iid  --local_bs 32 --local_ep 5 --account2

# Secagg
python3  main.py --gpu 0 --dataset cifar_10 --model  squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.001 --iid  --local_bs 32 --local_ep 5 --account2

# dp-nsa 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 1 --dp_sample 1 --dp_clip 5  --lr 0.001 --iid  --local_bs 32 --local_ep 5 --account2