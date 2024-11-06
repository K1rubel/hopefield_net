import numpy as np
# to imitate the idea raised during the training -- snippets of the orginal data
def add_horizontal_band_noise(image, band_height=2, num_bands=4):
    distorted_image = image.copy()
    img_height, img_width = image.shape
    
    # Calculate positions for horizontal bands
    band_spacing = img_height // (num_bands + 1)
    
    # Insert black bands (pixel value of -1) at calculated intervals
    for i in range(1, num_bands + 1):
        start_row = i * band_spacing
        end_row = min(start_row + band_height, img_height)
        distorted_image[start_row:end_row, :] = -1  # Set band to black (-1)
    
    return distorted_image