export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH=.

python -m torch.distributed.launch --nproc_per_node=2 --master_port 6015 main_supcon.py \
    --syncBN \
    --epochs 100 \
    --learning_rate 0.5 \
    --temp 0.5 \
    --cosine \
    --method SimCLR \
    --ngpu 2 



