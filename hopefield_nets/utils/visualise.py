import matplotlib.pyplot as plt

def visualize_results(original, distorted, retrieved):
    fig, axes = plt.subplots(3, len(original), figsize=(10, 4))
    titles = ["Original", "Segmented", "Recovered"]
    for i in range(len(original)):
        axes[0, i].imshow(original[i], cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(distorted[i], cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(retrieved[i], cmap='gray')
        axes[2, i].axis('off')
    for i in range(len(titles)):
        axes[i,0].set_title(titles[i])
    plt.show()