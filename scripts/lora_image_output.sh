#!/bin/bash

prompts=("prompt1" "prompt2" "prompt3")

xs=("test")

models=("test")

if [ ${#xs[@]} -ne ${#models[@]} ]; then
    echo "Error: xs and models arrays must have the same length."
    exit 1
fi

for i in "${!xs[@]}"
do
    x="${xs[$i]}"
    model_version="${models[$i]}"

    lora_id="./lora/${model_version}"

    for prompt in "${prompts[@]}"
    do
        output_dir="./result/lora/$x"

        if [ -d "$output_dir" ]; then
            echo "Directory $output_dir already exists. Skipping this combination."
            continue
        fi

        echo "Generating images for prompt: $prompt with model: ${model_version} in directory: $output_dir"
        python ./generate/lora_image_output.py --output_dir "$output_dir" --prompt "$prompt" --lora_id "$lora_id" --num "100"
        echo "Images generated for prompt '$prompt' with model ${model_version} in directory '$output_dir'."
    done
done

echo "All combinations of X values, models, and prompts processed."

