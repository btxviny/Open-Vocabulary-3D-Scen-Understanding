import cv2
import numpy as np

# Read the RGB image and the binary mask image
rgb_image = cv2.imread("rgb_image.png")
mask = cv2.imread("mask.png")

# Create a binary mask: non-black pixels are 1, black pixels are 0
binary_mask = np.all(mask != [0, 0, 0], axis=-1)  # True for non-black pixels, False for black pixels
binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)

# Perform connected components analysis
num_labels, labels = cv2.connectedComponents(binary_mask_uint8)

# Define the minimum component size (remove small components)
min_size = 500  # Minimum size of component to keep (in pixels)

# Calculate the size of each component
component_sizes = np.bincount(labels.ravel())[1:]  # Exclude background (label 0)

# Create an empty mask for the large components
large_components_mask = np.zeros_like(binary_mask_uint8)

# Iterate over the components (starting from 1 to avoid background)
for label in range(1, num_labels):
    if component_sizes[label - 1] >= min_size:  # Only process large components
        # Add the current component's mask to the large_components_mask
        large_components_mask[labels == label] = 255  # Set the pixels corresponding to the component to 255

# Convert the large_components_mask to 3 channels for visualization if needed
large_components_mask_rgb = cv2.cvtColor(large_components_mask, cv2.COLOR_GRAY2BGR)

# Display the large components binary mask
cv2.imshow("Large Components Mask", large_components_mask)
cv2.imshow("Large Components Mask (RGB)", large_components_mask_rgb)  # Show the mask in RGB format
cv2.waitKey(5)
cv2.destroyAllWindows()

# Optionally, save the final binary mask
cv2.imwrite("large_components_mask.png", large_components_mask)
