import os
import cv2
from segmentation_module import NucleusSegmentation

def augment_image(image):
    augmented_images = []
    augmented_images.append(image)
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE))
    augmented_images.append(cv2.rotate(image, cv2.ROTATE_180))
    augmented_images.append(cv2.flip(image, 0))  # Vertical flip
    augmented_images.append(cv2.flip(image, 1))  # Horizontal flip
    return augmented_images

def preprocess_data(input_folder, output_folder):
    # Initialize the NucleusSegmentation class
    segmenter = NucleusSegmentation(min_area=100)
    
    for file_name in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, file_name))
        augmented_images = augment_image(image)
        
        for i, aug_img in enumerate(augmented_images):
            Nucleus_img, img_convex, img_ROC = segmenter.segmentation(aug_img)
            output_path = os.path.join(output_folder, f"{file_name}_segmented_{i}.png")
            cv2.imwrite(output_path, Nucleus_img)

if __name__ == "__main__":
    input_images_root = "path to original images"
    processed_images_root = "path to save preprocessed images"
    preprocess_data(input_images_root, processed_images_root)
