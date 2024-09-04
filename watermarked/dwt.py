import os
import numpy as np
import pywt
from PIL import Image

def apply_dwt_watermark_color(image_path, output_path, watermark):
    img = Image.open(image_path).convert('RGB')
    data = np.array(img)
    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]

    channels = []
    for color in (red, green, blue):
        coeffs = pywt.dwt2(color, 'haar')
        cA, (cH, cV, cD) = coeffs

        watermark_resized = np.resize(watermark, cH.shape)

        cH += watermark_resized
        new_coeffs = cA, (cH, cV, cD)
        watermarked_channel = pywt.idwt2(new_coeffs, 'haar')
        channels.append(watermarked_channel)

    watermarked_image = np.stack(channels, axis=-1)
    Image.fromarray(np.uint8(watermarked_image.clip(0, 255))).save(output_path)

def process_images(source_dir, target_dir, watermark):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(source_dir):
        if filename.endswith('.jpg'):
            source_path = os.path.join(source_dir, filename)
            target_path = os.path.join(target_dir, filename)
            apply_dwt_watermark_color(source_path, target_path, watermark)
            print(f"Processed {filename}")

watermark = np.random.rand(256, 256) * 10

input_folder = "./data/origin"
output_folder = "./data/test"

process_images(input_folder, output_folder, watermark)
