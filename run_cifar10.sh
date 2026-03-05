# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9


# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 0 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9

# python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
# python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9


python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn  --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9


# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10
# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9

# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10
# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 1 --dataset cifar_10 --model squeezenet --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9

# python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10
# python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
# python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9

python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10
python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 1 --dataset cifar_10 --model cifarcnn  --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 0.3 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9

