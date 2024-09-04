#!/bin/bash

export MODEL_NAME="./base_diffusion_model/???"
export OUTPUT_DIR="./tmp"

xs=("test")

pgd_eps_values=("0.05")

for keyword in "${xs[@]}"
do
    for eps in "${pgd_eps_values[@]}"
    do
        echo "Processing keyword: $keyword with pgd_eps: $eps"
        export DATASET_NAME="./data/origin"
        save_path="./data/adv_result/${keyword}/adv-${eps}"
        
        accelerate launch ./train/train_text_to_image_adv.py \
          --save_addr="$save_path" \
          --mixed_precision="fp16" \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --train_data_dir=$DATASET_NAME \
          --output_dir=${OUTPUT_DIR} \
          --seed=1337 \
          --checkpointing_steps=200 \
          --snr_gamma=5.0 \
          --image_column="image" \
          --caption_column="additional_feature" \
          --max_train_steps=10000 \
          --lr_scheduler="cosine" \
          --lr_warmup_steps=0 \
          --train_batch_size=2 \
          --pgd_eps=$eps \
          --gradient_accumulation_steps=2 \

        echo "Finished processing keyword: $xs with pgd_eps: $eps"
    done
done

echo "All keywords processed."


