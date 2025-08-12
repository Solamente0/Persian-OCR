#!/usr/bin/env python3
"""
Persian OCR Usage Script
This script demonstrates how to use the Persian-OCR library for recognizing Persian letters in images.
Based on the repository: https://github.com/Solamente0/Persian-OCR
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
# import torch  # If using PyTorch instead of TensorFlow
import matplotlib.pyplot as plt

class PersianOCR:
    def __init__(self, model_path=None):
        """
        Initialize the Persian OCR system
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.model_path = model_path
        
        # Persian alphabet mapping (you'll need to adjust based on the actual model)
        self.persian_chars = [
            'ا', 'ب', 'پ', 'ت', 'ث', 'ج', 'چ', 'ح', 'خ', 'د',
            'ذ', 'ر', 'ز', 'ژ', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ',
            'ع', 'غ', 'ف', 'ق', 'ک', 'گ', 'ل', 'م', 'ن', 'و',
            'ه', 'ی'
        ]
        
        if model_path:
            self.load_model()
    
    def load_model(self):
        """Load the trained Persian OCR model"""
        try:
            # For TensorFlow/Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
            
            # For PyTorch model (alternative)
            # self.model = torch.load(self.model_path)
            # self.model.eval()
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        return True
    
    def preprocess_image(self, image_path):
        """
        Preprocess the input image for Persian OCR
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to model input size (typically 28x28 for LeNet5)
        resized = cv2.resize(gray, (28, 28))
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Reshape for model input (add batch and channel dimensions)
        processed = normalized.reshape(1, 28, 28, 1)
        
        return processed
    
    def segment_text(self, image_path):
        """
        Segment text into individual characters
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of segmented character images
        """
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract character regions
        characters = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small contours
            if w > 10 and h > 10:
                char_img = gray[y:y+h, x:x+w]
                characters.append((x, char_img))  # Store x-coordinate for sorting
        
        # Sort characters by x-coordinate (right-to-left for Persian)
        characters.sort(key=lambda x: x[0], reverse=True)
        
        return [char[1] for char in characters]
    
    def predict_character(self, char_image):
        """
        Predict a single Persian character
        
        Args:
            char_image (numpy.ndarray): Character image
            
        Returns:
            str: Predicted Persian character
        """
        if self.model is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        # Preprocess the character image
        processed_img = self.preprocess_image(char_image)
        
        # Make prediction
        prediction = self.model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Map prediction to Persian character
        if predicted_class < len(self.persian_chars):
            return self.persian_chars[predicted_class]
        else:
            return '?'
    
    def recognize_text(self, image_path):
        """
        Recognize Persian text in an image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Recognized Persian text
        """
        # Segment the image into characters
        characters = self.segment_text(image_path)
        
        # Recognize each character
        recognized_text = ""
        for char_img in characters:
            char = self.predict_character(char_img)
            recognized_text += char
        
        return recognized_text
    
    def display_results(self, image_path, recognized_text):
        """
        Display the original image and recognized text
        
        Args:
            image_path (str): Path to the input image
            recognized_text (str): Recognized text
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.text(0.1, 0.5, recognized_text, fontsize=20, ha='left', va='center')
        plt.title('Recognized Text')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function demonstrating how to use the Persian OCR system
    """
    # Initialize the OCR system
    # You'll need to provide the path to your trained model
    model_path = "./LeNet5.h5"  # or .pth for PyTorch
    
    ocr = PersianOCR(model_path)
    
    # Path to your input image containing Persian text
    image_path = "./text_image.jpg"
    
    try:
        # Recognize text in the image
        recognized_text = ocr.recognize_text(image_path)
        print(f"Recognized text: {recognized_text}")
        
        # Display results
        ocr.display_results(image_path, recognized_text)
        
    except Exception as e:
        print(f"Error during OCR processing: {e}")

# Example usage without a trained model (for testing preprocessing)
def test_preprocessing():
    """
    Test the preprocessing functions without a trained model
    """
    ocr = PersianOCR()  # No model loaded
    
    # Test image preprocessing
    image_path = "./test_image.jpg"
    
    try:
        # Segment characters
        characters = ocr.segment_text(image_path)
        print(f"Found {len(characters)} character segments")
        
        # Display segmented characters
        fig, axes = plt.subplots(1, min(len(characters), 10), figsize=(15, 3))
        for i, char_img in enumerate(characters[:10]):
            if len(characters) == 1:
                axes.imshow(char_img, cmap='gray')
                axes.set_title(f'Char {i+1}')
                axes.axis('off')
            else:
                axes[i].imshow(char_img, cmap='gray')
                axes[i].set_title(f'Char {i+1}')
                axes[i].axis('off')
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error during preprocessing test: {e}")

if __name__ == "__main__":
    # Run the main OCR function
    main()
    
    # Or test preprocessing only
    # test_preprocessing()
