import numpy as np
from tensorflow.keras.datasets import mnist

# load data from MNIST and binarize into -1 1 (pixel > 127 --1 else -1)

def load_and_binarize_image():
    (train_images, train_labels), (_, _) = mnist.load_data()
    
    # Select a subset of 10 images(0-9)
    selected_images = []
    for digit in range(10):
        index = np.where(train_labels == digit)[0][0]
        selected_images.append(train_images[index])
    
    # Binarize images to -1 (black) and 1 (white)
    binarized_images = [np.where(image > 127, 1, -1) for image in selected_images]
    return np.array(binarized_images)

