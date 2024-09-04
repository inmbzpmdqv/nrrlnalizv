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
    export OUTPUT_DIR="./ST/$y"
 
    echo "Training with dataset $x and outputting to $y..."
    accelerate launch ./train/train_text_to_image_ST.py \
        --mixed_precision="fp16" \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --train_data_dir=$DATASET_NAME \
        --output_dir=$OUTPUT_DIR \
        --seed=1337 \
        --checkpointing_steps=5000 \
        --snr_gamma=5.0 \
        --image_column="image" \
        --caption_column="additional_feature" \
        --max_train_steps=10000 \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=0 \
        --train_batch_size=2 \
        --gradient_accumulation_steps=2
done

echo "All training tasks completed."

