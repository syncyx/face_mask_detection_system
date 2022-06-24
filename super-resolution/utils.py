import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from PIL import Image


def load_image(path):
    return np.array(Image.open(path))

def save_image(path, img):
    pil_img = tf.keras.preprocessing.image.array_to_img(img)
    pil_img.save(path)
    return path

def plot_sample(lr, sr):
    plt.figure(figsize=(20, 10))

    images = [lr, sr]
    titles = ['LR', f'SR (x{sr.shape[0] // lr.shape[0]})']

    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 2, i+1)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
