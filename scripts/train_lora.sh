#!/bin/bash

xs=("test")
ys=("test")

if [ ${#xs[@]} -ne ${#ys[@]} ]; then
    echo "Error: The length of xs and ys arrays must be the same."
    exit 1
fi

for i in "${!xs[@]}"; do
    x=${xs[$i]}
    y=${ys[$i]}

    export MODEL_NAME="./base_diffusion_model/???"

    export DATASET_NAME="./data/$x"
    export OUTPUT_DIR="./lora/$y"

    echo "Training with dataset $x and outputting to $y..."
    accelerate  launch ./train/train_text_to_image_lora.py \
    --mixed_precision="fp16" \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_NAME \
    --dataloader_num_workers=8 \
    --resolution=512 \
    --center_crop \
    --random_flip \
    --train_batch_size=2 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=10000 \
    --image_column="image" \
    --caption_column="additional_feature" \
    --learning_rate=1e-04 \
    --max_grad_norm=1 \
    --lr_scheduler="cosine" \
    --lr_warmup_steps=0 \
    --output_dir=${OUTPUT_DIR} \
    --checkpointing_steps=23000 \
    --seed=1337
done

echo "All training tasks completed."

