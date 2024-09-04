
from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating images")
parser.add_argument("--model_id", type=str, required=True, help="Model ID or path for loading the model")
parser.add_argument("--num", type=int, required=True, help="the number of pictures need to gen")

args = parser.parse_args()
pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

os.makedirs(args.output_dir, exist_ok=True)


num_images = args.num


for i in range(num_images):
    image = pipe(args.prompt).images[0]
    image_number = str(i).zfill(3)
    image.save(os.path.join(args.output_dir, f"{image_number}.png"))

print(f"Generated and saved {num_images} images to {args.output_dir}.")