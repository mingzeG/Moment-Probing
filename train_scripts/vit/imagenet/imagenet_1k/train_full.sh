CUDA_VISIBLE_DEVICES=0,1,2,3  python  -m torch.distributed.launch --nproc_per_node=4 --master_port=33518 \
	train.py /path/to/ILSVRC2012/images --dataset imagenet --num-classes 1000 --model vit_base_patch16_224_in21k \
    --batch-size 64 --epochs 30 \
	--opt adamw --weight-decay 0.05 \
    --warmup-lr 1e-7 --warmup-epochs 5  \
    --lr 1e-4 --min-lr 1e-8 \
    --drop-path 0.2 --img-size 224 \
	--model-ema --model-ema-decay 0.99992  \
	--output  output/vit_base_patch16_224_in21k/imagenet_1k/full \
	--amp  --pretrained  \