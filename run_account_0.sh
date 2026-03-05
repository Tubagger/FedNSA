# #cifar10
# 100clients
# # Niss
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 1 --account

# # Chain-PPFL
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.001 --iid  --local_bs 32 --local_ep 5 --account

# # Secagg
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account

# # dp-nsa 
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam --lr 0.001 --iid  --local_bs 32 --local_ep 5 --account

# 50clients
# Niss
python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5  --optim Adam --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account

# # Chain-PPFL
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Chain-PPFL --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account

# # Secagg
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm Secagg --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 50 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account

# # dp-nsa 
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 500 --dp_sample 1 --dp_clip 5 --optim Adam  --lr 0.0001 --iid  --local_bs 32 --local_ep 5 --account
