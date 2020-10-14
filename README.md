# SimCLR_pytorch_distributed

This repository is modified from [SupContrast](https://github.com/HobbitLong/SupContrast) to support distributed training of SimCLR with syncBN. Here we only modify the code to train on two NVIDIA 2080Ti GPUs with batch size 256 on CIFAR-10/CIFAR-100 datasets.

## Requirements

This repo was tested with Ubuntu 16.04.1 LTS, Python 3.6, PyTorch 1.2.0, and CUDA 10.1. 

## Training

1. Pretraining stage.

   ```
   CUDA_VISIBLE_DEVICES=0,1 \
   python -m torch.distributed.launch --nproc_per_node=2 --master_port 6015 main_supcon.py \
     --syncBN \
     --epochs 100 \
     --learning_rate 0.5 \
     --temp 0.5 \
     --cosine \
     --method SimCLR \
     --ngpu 2 
   ```

   You can modify ```--master_port``` to assign another available port. You can also train the model with ```sh run_supcon.sh```.

2. Linear evaluation stage.

   ```
   CUDA_VISIBLE_DEVICES=0 python main_linear.py \
     --learning_rate 5 \
     --batch_size 256 \
     --ckpt path/to/ckpt
   ```

   You can also train the model with ```sh run_linear.sh```. The checkpoints are stored in ```./work_space```.

## Results

Performance on CIFAR-10:

|        |   Arch    | Head |  BS  | Embedding Dim | Training Epoch | Top-1 (%) | Top-5 (%) |
| :----: | :-------: | :--: | :--: | :-----------: | :------------: | :-------: | :-------: |
| SimCLR | ResNet-50 | MLP  | 256  |      128      |      100       |   84.76   |   99.36   |
| SimCLR | ResNet-50 | MLP  | 256  |      128      |      200       |   89.05   |   99.69   |

and on CIFAR-100:

|        |   Arch    | Head |  BS  | Embedding Dim | Training Epoch | Top-1 (%) | Top-5 (%) |
| :----: | :-------: | :--: | :--: | :-----------: | :------------: | :-------: | :-------: |
| SimCLR | ResNet-50 | MLP  | 256  |      128      |      100       |   58.43   |   85.26   |
| SimCLR | ResNet-50 | MLP  | 256  |      128      |      200       |   65.73   |   89.64   |

