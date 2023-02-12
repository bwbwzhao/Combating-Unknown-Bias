# Combating Unknown Bias with Effective Bias-Conflicting Scoring and Gradient Alignment

## Installation
```
pip install -r requirements.txt
```

## Datasets
There are five datasets used in this repository: Colored MNIST, Corrupted CIFAR-10<sup>1</sup>, Corrupted CIFAR-10<sup>2</sup>, Biased Waterbirds and Biased CelebA. Please prepare these datasets as following.

- For Colored MNIST, Corrupted CIFAR-10<sup>1</sup> and Corrupted CIFAR-10<sup>2</sup>, please run 
    ```
    python3 ./data/make_dataset.py
    ```
    Then, the directory structures will be as following:
    ```
    debias_datasets
    ├── mnist
    │   ├── ColoredMNIST-Skewed0.005-Severity4
    │   ├── ColoredMNIST-Skewed0.01-Severity4
    │   ├── ColoredMNIST-Skewed0.02-Severity4
    │   ├── ColoredMNIST-Skewed0.05-Severity4
    │   └── MNIST
    └── cifar10
        ├── CorruptedCIFAR10-Type0-Skewed0.005-Severity4
        ├── CorruptedCIFAR10-Type0-Skewed0.01-Severity4
        ├── CorruptedCIFAR10-Type0-Skewed0.02-Severity4
        ├── CorruptedCIFAR10-Type0-Skewed0.05-Severity4
        ├── CorruptedCIFAR10-Type1-Skewed0.005-Severity4
        ├── CorruptedCIFAR10-Type1-Skewed0.01-Severity4
        ├── CorruptedCIFAR10-Type1-Skewed0.02-Severity4
        ├── CorruptedCIFAR10-Type1-Skewed0.05-Severity4
        └── CIFAR10
    ```

- For Biased Waterbirds, please download it following https://github.com/kohpangwei/group_DRO .
    ```
    debias_datasets
    └── waterbird
        ├── img
        │   ├── 001.Black_footed_Albatross
        │   │   ├── Black_Footed_Albatross_0001_796111.jpg
        │   │   ├── ...
        │   │   └── Black_Footed_Albatross_0090_796077.jpg
        │   ├── 002.Laysan_Albatross
        │   ├── ...
        │   ├── 199.Winter_Wren
        │   └── 200.Common_Yellowthroat
        ├── train.txt
        └── valid.txt
    ```

- For Biased CelebA, please download it following http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html .
    ```
    debias_datasets
    └── celeba
        ├── img_align_celeba
        │   ├── 000001.jpg
        │   ├── ...
        │   └── 202599.jpg
        ├── list_attr_celeba.txt
        ├── list_bbox_celeba.txt
        ├── list_landmarks_align_celeba.txt
        ├── identity_CelebA.txt
        └── list_eval_partition.txt 
    ```


## Training

The debiasing methods are provided in `./debiasing/` and complete scripts are provided in `./scripts/`. Here, we present the examples for Colored MNIST ($\rho=0.99$).

- Vanilla
```
python3 ./debiasing/main_vanilla.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'vanilla'
```

- GA
```
python3 ./debiasing/main_ga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'ga' --gamma 1.6
```

- ECS+GA
```
python3 ./mining/det_peer.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'det_peer' --eta 0.5

python3 ./debiasing/main_ecsga.py --cfg './configs/mnist.yaml' --dataset_tag 'ColoredMNIST-Skewed0.01-Severity4' --main_tag 'ecsga' --gamma 1.6 --bmodel 'det_peer_eta0.5' 
```

The trained models and logs will be saved in `./results/`.

## Acknowledgment
The code is greatly inspired by (heavily from) the [LfF](https://github.com/alinlab/LfF).

