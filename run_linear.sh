export CUDA_VISIBLE_DEVICES=0

PYTHONPATH=. python main_linear.py \
    --learning_rate 5 \
    --batch_size 256 \
    --ckpt path/to/ckpt

