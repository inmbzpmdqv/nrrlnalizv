from diffusers import StableDiffusionPipeline
import torch
import os
import argparse

parser = argparse.ArgumentParser(description="Generate images with Stable Diffusion")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
parser.add_argument("--prompt", type=str, required=True, help="Prompt for generating images")
parser.add_argument("--lora_id", type=str, required=True, help="Lora ID or path for loading the model")
parser.add_argument("--num", type=int, required=True, help="the number of pictures need to gen")

args = parser.parse_args()

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

model_path = "./base_diffusion_model/{???}"

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
lora_model_path=args.lora_id
pipe.load_lora_weights(lora_model_path, weight_name="pytorch_lora_weights.safetensors")
pipe = pipe.to("cuda")

os.makedirs(args.output_dir, exist_ok=True)

num_images = args.num

for i in range(num_images):
    lora_scale = 1.2
    image = pipe(
        args.prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}
    ).images[0]
    image_number = str(i).zfill(3)
    image.save(os.path.join(args.output_dir, f"{image_number}.png"))

print(f"Generated and saved {num_images} images to {args.output_dir}.")

