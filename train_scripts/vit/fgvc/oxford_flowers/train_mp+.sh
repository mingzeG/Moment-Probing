CUDA_VISIBLE_DEVICES=0,1,2,3  python -m torch.distributed.launch --nproc_per_node=4  --master_port=12341 \
    train.py /path/to/oxford_flowers  --dataset torch/oxford_flowers --num-classes 102 --val-split val --simple-aug --model vit_base_patch16_224_in21k  \
    --batch-size 16 --epochs 100 \
	--opt adamw  --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 10  \
    --lr 1e-2 --min-lr 1e-8 \
    --drop-path 0 --img-size 224 \
	--output  output/vit_base_patch16_224_in21k/oxford_flowers/mp+ \
	--amp --tuning-mode psrp --probing-mode mp --pretrained  \