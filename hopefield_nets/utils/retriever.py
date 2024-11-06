import numpy as np

def retrieve(weights, noisy_image, steps):
    # match dimension of the recovered image with the noisy one

    recovered_image = noisy_image.reshape(-1)
    # make use of the sum(w_ij * x_i)  and update the sate accordingly
    for _ in range(steps): 
        recovered_image = np.sign(np.dot(weights, recovered_image))

    return recovered_image.reshape(28, 28)