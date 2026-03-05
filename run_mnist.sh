### model lenet5
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 


python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9


python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9


python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9

python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 0 --dataset mnist --model lenet5 --algorithm NISS --dp_mechanism MA --dp_epsilon 1 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9

### model cnn
python3  main.py --gpu 0 --dataset mnist --model mnistcnn --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 
python3  main.py --gpu 0 --dataset mnist --model mnistcnn --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 

python3  main.py --gpu 0 --dataset mnist --model mnistcnn --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 
python3  main.py --gpu 0 --dataset mnist --model mnistcnn --algorithm DP-NSA --dp_mechanism no_dp --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 


python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7 
python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9 


python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7  
python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 10 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9 


python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism NA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --iid  --local_bs 8 --local_ep 10 --k 0.9

python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.7
python3  main.py --gpu 1 --dataset mnist --model mnistcnn --algorithm NISS --dp_mechanism MA --dp_epsilon 5 --dp_delta 1e-5 --num_users 100 --epochs 50 --dp_sample 1 --dp_clip 3  --lr 0.01 --alpha 0.3  --local_bs 8 --local_ep 10 --k 0.9
