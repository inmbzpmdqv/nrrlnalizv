#!/bin/bash

prompts=("prompt1" "prompt2" "prompt3")

xs=("test")
 

models=("test")

if [ ${#models[@]} -ne ${#xs[@]} ]; then
    echo "Error: The length of the models array must be equal to the length of xs array."
    exit 1
fi


for x in "${xs[@]}"; 
do
    model_version="${models[$i]}"
    model_id="./ST/${model_version}"
    for prompt in "${prompts[@]}"; do
        output_dir="./result/ST/$x"
        echo "Generating images for prompt: $prompt with model: ${model_version} in directory: $output_dir"
        sudo python ./generate/ST_image_output.py --output_dir "$output_dir" --prompt "$prompt" --model_id "$model_id" --num "100"
        echo "Images generated for prompt '$prompt' with model ${model_version} in directory '$output_dir'."
    done
done

echo "All combinations of dataset, X values, and prompts processed."                                                                      


