import os
import numpy as np
from PIL import Image


def apply_gaussian_watermark(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    data = np.array(img)
    mean = 0
    std = 5
    gaussian_noise = np.random.normal(mean, std, data.shape)
    watermarked_data = data + gaussian_noise
    watermarked_data = np.clip(watermarked_data, 0, 255)
    Image.fromarray(np.uint8(watermarked_data)).save(output_path)

def process_images(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            apply_gaussian_watermark(source_path, target_path)
            print(f"Processed {filename}")


input_folder = "./data/origin"
output_folder = "./data/test"

process_images(input_folder, output_folder)
