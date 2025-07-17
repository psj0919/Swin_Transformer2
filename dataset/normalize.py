import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

if __name__ == '__main__':
    img_dir = '/storage/sjpark/vehicle_data/Dataset/train_image'
    img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir) if fname.endswith(('.png', '.jpg'))]

    mean = np.zeros(3)
    std = np.zeros(3)
    n_pixels = 0

    for path in tqdm(img_paths):
        img = Image.open(path).convert('RGB')
        img = np.array(img).astype(np.float32) / 255.0

        mean += img.mean(axis=(0, 1))
        std += img.std(axis=(0, 1))
        n_pixels += 1


    mean /= n_pixels
    std /= n_pixels

    print('Mean:', mean)
    print('Std:', std)

Mean: [0.48123408, 0.4961526,  0.50759056]
Std: [0.16502722, 0.17250277, 0.19733038]
