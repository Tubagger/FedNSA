#mnist
# 100clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0

# 50clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1  --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 50 --dp_sample 1 --dp_clip 5  --lr 0.01 --iid  --local_bs 256 --local_ep 1 --account --acc 98.0

# #cifar10
# 100clients
# Niss
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account--acc 78.0

# 50clients
# Niss
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account --acc 78.0

# 15clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 25clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 35clients
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 100 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10  --account --acc 95.0 

# 15clients
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 15 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
# 25clients
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 25 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
# 35clients
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0 
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 35 --epochs 300 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 5 --account --acc 65.0
