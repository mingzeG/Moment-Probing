CUDA_VISIBLE_DEVICES=0,  python validate_ood.py \
    /path/to/imagenet-c/  \
    --num-classes 1000 \
    --model vit_base_patch16_224_in21k \
    --batch-size 64 \
    --imagenet_c \
	--results-file  output/vit_base_patch16_224_in21k/imagenet_c/mp \
    --tuning-mode linear_probe \
    --checkpoint /path/to/vit_base_patch16_224_in21k/imagenet_1k/mp/model_best.pth.tar
