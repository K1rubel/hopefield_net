import numpy as np
# adds random noise parts to the image based on certain distribution function or defined masks(contigious regions of white pixels) 
def corrupt_input(image_to_corrupt, method, bernouli_noise_probability = 0.1, mask_value = -1, mask_matrix=np.zeros((2,2))):
    if method == 'bernouli':
        noise_image = (-1*np.random.binomial(1, bernouli_noise_probability, (image_to_corrupt.shape[0], image_to_corrupt.shape[1]))).astype(np.float32)
        noise_image[noise_image==0] = 1
        corrupted_image = noise_image * image_to_corrupt
    elif method == 'mask':
        corrupted_image = np.copy(image_to_corrupt)
        corrupted_image[mask_matrix[0,0]:mask_matrix[0,1], 
                        mask_matrix[1,0]:mask_matrix[1,1]] = mask_value

    return corrupted_image 