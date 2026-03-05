# dp-nsa 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1  

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1  

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 


# NIss
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.9

python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.9

python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.9

python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.9


python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --iid  --local_bs 128 --local_ep 1 --k 0.9

python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism Gaussian --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 150 --dp_sample 1 --dp_clip 3  --lr 0.1 --alpha 0.1  --local_bs 128 --local_ep 1 --k 0.9
