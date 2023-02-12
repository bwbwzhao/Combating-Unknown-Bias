# ============================================ vanilla ===================================
seed=0

python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'vanilla' --seed $seed &
wait

python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.003-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.001-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.0005-Severity4' --main_tag 'vanilla' --seed $seed &
wait

python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'vanilla' --seed $seed &
wait

python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'vanilla' --seed $seed &
python3 ./debiasing/main_vanilla.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'vanilla' --seed $seed &
wait

python3 ./debiasing/main_vanilla.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'vanilla' --seed $seed
python3 ./debiasing/main_vanilla.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'vanilla' --seed $seed


# ============================================ rew ===================================
seed=0

python3 ./debiasing/main_rew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_rew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'rew' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_rew.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'rew' --seed $seed
python3 ./debiasing/main_rew.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'rew' --seed $seed





# ============================================ lff ===================================
seed=0

python3 ./debiasing/main_lff.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'lff' --seed $seed & 
wait

python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'lff' --seed $seed & 
wait

python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'lff' --seed $seed & 
python3 ./debiasing/main_lff.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'lff' --seed $seed & 
wait

python3 ./debiasing/main_lff.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'lff' --seed $seed 
python3 ./debiasing/main_lff.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'lff' --seed $seed 




# ============================================ ga ===================================
seed=0

python3 ./debiasing/main_ga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
python3 ./debiasing/main_ga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'ga' --gamma 1.6 --seed $seed &
wait

python3 ./debiasing/main_ga.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'ga' --seed $seed
python3 ./debiasing/main_ga.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'ga' --seed $seed



# ============================================ mining ===================================
seed=0

python3 ./mining/det_peer.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'det_peer' --eta 0.5 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'det_peer' --eta 0.5 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'det_peer' --eta 0.5 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'det_peer' --eta 0.5 --seed $seed &
wait

python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
wait

python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
python3 ./mining/det_peer.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'det_peer' --eta 0.1 --seed $seed &
wait

python3 ./mining/det_peer.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'det_peer' --eta 0.9 --seed $seed
python3 ./mining/det_peer.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'det_peer' --eta 0.9 --seed $seed


# ============================================ ecs+rew ===================================
seed=0

python3 ./debiasing/main_ecsrew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed & 
python3 ./debiasing/main_ecsrew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &
python3 ./debiasing/main_ecsrew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &
python3 ./debiasing/main_ecsrew.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &

wait

python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed & 
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
wait

python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed & 
python3 ./debiasing/main_ecsrew.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'ecsrew' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
wait

python3 ./debiasing/main_ecsrew.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'ecsrew' --gamma 1.0 --bmodel 'det_peer_eta0.9' --seed $seed
python3 ./debiasing/main_ecsrew.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'ecsrew' --gamma 1.0 --bmodel 'det_peer_eta0.9' --seed $seed


# ============================================ ecs+ga ===================================
seed=0

python3 ./debiasing/main_ecsga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.005-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed & 
python3 ./debiasing/main_ecsga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &
python3 ./debiasing/main_ecsga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.02-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &
python3 ./debiasing/main_ecsga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.05-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'L_det_peer_eta0.5' --seed $seed &
wait

python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.005-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.01-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.02-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed & 
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type0-Skewed0.05-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
wait

python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.005-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.01-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.02-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed & 
python3 ./debiasing/main_ecsga.py --cfg './configs/cifar.yaml' --dataset_tag 'CorruptedCIFAR10-Type1-Skewed0.05-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.1' --seed $seed &  
wait

python3 ./debiasing/main_ecsga.py --cfg './configs/bird.yaml' --dataset_tag 'BIRD' --main_tag 'ecsga' --gamma 1.0 --bmodel 'det_peer_eta0.9' --seed $seed
python3 ./debiasing/main_ecsga.py --cfg './configs/celeba.yaml' --dataset_tag 'CelebA' --main_tag 'ecsga' --gamma 1.0 --bmodel 'det_peer_eta0.9' --seed $seed


