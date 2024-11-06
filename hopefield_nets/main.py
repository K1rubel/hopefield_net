
from utils import load_and_binarize_image, train_hopfield_weights, corrupt_input, retrieve, add_horizontal_band_noise, visualize_results

import matplotlib.pyplot as plt
import numpy as np

images = load_and_binarize_image()
# select image 0
n = images.shape[0]
iterations = 100
original = []
segmented = [] 
# noisy_image = []
retrieved = [] 
for i in range(n):
    neurons = images[i].shape[0] * images[i].shape[1]
    weights = np.zeros((neurons, neurons)).astype(np.float32)
    weights = train_hopfield_weights(images[i], weights)
    original.append(images[i])

    segmented.append(add_horizontal_band_noise(images[i], band_height=5, num_bands=3))
    retrieved.append(retrieve(weights,segmented[i], iterations))

visualize_results(original, segmented, retrieved)
# img = images[7]

# plt.imshow(img, interpolation='None', cmap='gray')
# plt.show()

# neurons = img.shape[0] * img.shape[1] #784 (28 * 28)
# weights = np.zeros((neurons, neurons)).astype(np.float32)

# weights = train_hopfield_weights(img, weights)

# corrupted_image = corrupt_input(img, method='bernouli', bernouli_noise_probability=0.1)

# segmented_image = add_horizontal_band_noise(img, band_height=5, num_bands=3)

# # view original and corrupted image

# # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,7.5))
# # ax[0].imshow(img, interpolation='None', cmap='gray')
# # ax[1].imshow(corrupted_image, interpolation='None', cmap='gray')
# # plt.show()

# iterations = 100

# retrieved_image = retrieve(weights,corrupted_image, iterations)
# retrieved_image_2 = retrieve(weights, segmented_image, iterations)

# fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize=(15, 7.5))
# ax[0].imshow(corrupted_image, interpolation='None', cmap='gray')
# ax[0].set_title('Corrupted Image -- burnoils noise')
# ax[1].imshow(retrieved_image, interpolation='None', cmap='gray')
# ax[1].set_title('Recovered Image 1')
# ax[2].imshow(segmented_image, interpolation='None', cmap='gray')
# ax[2].set_title('Corrupted Image -- segmentation')
# ax[3].imshow(retrieved_image_2, interpolation='None', cmap='gray')
# ax[3].set_title('Recovered Image 2')
# plt.show()