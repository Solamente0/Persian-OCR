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
        print(f"🔄 Loading model from: {self.model_path}")
        try:
            # For TensorFlow/Keras model
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Model loaded successfully from {self.model_path}")
            print(f"📊 Model input shape: {self.model.input_shape}")
            print(f"📊 Model output shape: {self.model.output_shape}")
            
            # For PyTorch model (alternative)
            # self.model = torch.load(self.model_path)
            # self.model.eval()
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
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
        print(f"🔄 Preprocessing image: {image_path if isinstance(image_path, str) else 'array input'}")
        
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            print(f"📐 Original image shape: {image.shape}")
        else:
            image = image_path
            print(f"📐 Input array shape: {image.shape}")
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("🎨 Converted to grayscale")
        else:
            gray = image
            print("🎨 Image already in grayscale")
        
        # Resize to model input size (typically 28x28 for LeNet5)
        resized = cv2.resize(gray, (28, 28))
        print(f"📏 Resized to: {resized.shape}")
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        print(f"🔢 Normalized pixel values (min: {normalized.min():.3f}, max: {normalized.max():.3f})")
        
        # Reshape for model input (add batch and channel dimensions)
        processed = normalized.reshape(1, 28, 28, 1)
        print(f"🔄 Final shape for model: {processed.shape}")
        
        return processed
    
    def segment_text(self, image_path):
        """
        Segment text into individual characters
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            list: List of segmented character images
        """
        print(f"🔍 Starting text segmentation for: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"📐 Image dimensions: {gray.shape}")
        
        # Apply threshold to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        print("🎯 Applied binary threshold using OTSU method")
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"🔍 Found {len(contours)} contours")
        
        # Extract character regions
        characters = []
        filtered_count = 0
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small contours
            if w > 10 and h > 10:
                char_img = gray[y:y+h, x:x+w]
                characters.append((x, char_img))  # Store x-coordinate for sorting
                print(f"✅ Character {filtered_count + 1}: position=({x},{y}), size=({w}x{h})")
                filtered_count += 1
            else:
                print(f"❌ Filtered out small contour {i + 1}: size=({w}x{h})")
        
        # Sort characters by x-coordinate (right-to-left for Persian)
        characters.sort(key=lambda x: x[0], reverse=True)
        print(f"📝 Sorted {len(characters)} characters for Persian (right-to-left) reading")
        
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
        
        print(f"🔮 Predicting character from image shape: {char_image.shape}")
        
        # Preprocess the character image
        processed_img = self.preprocess_image(char_image)
        
        # Make prediction
        print("🤖 Running model prediction...")
        prediction = self.model.predict(processed_img, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        print(f"📊 Prediction confidence: {confidence:.4f}")
        print(f"🏷️ Predicted class index: {predicted_class}")
        
        # Map prediction to Persian character
        if predicted_class < len(self.persian_chars):
            predicted_char = self.persian_chars[predicted_class]
            print(f"📝 Predicted character: '{predicted_char}'")
            return predicted_char
        else:
            print(f"❓ Unknown class index {predicted_class}, returning '?'")
            return '?'
    
    def recognize_text(self, image_path):
        """
        Recognize Persian text in an image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Recognized Persian text
        """
        print(f"\n🚀 Starting Persian text recognition for: {image_path}")
        print("=" * 60)
        
        # Segment the image into characters
        characters = self.segment_text(image_path)
        print(f"\n🔤 Processing {len(characters)} characters...")
        
        # Recognize each character
        recognized_text = ""
        for i, char_img in enumerate(characters):
            print(f"\n--- Character {i + 1}/{len(characters)} ---")
            char = self.predict_character(char_img)
            recognized_text += char
            print(f"✅ Added '{char}' to result")
        
        print(f"\n🎉 Final recognized text: '{recognized_text}'")
        print("=" * 60)
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
    print("🎯 Persian OCR System - Starting Main Function")
    print("=" * 50)
    
    # Initialize the OCR system
    # You'll need to provide the path to your trained model
    model_path = "path/to/your/trained_model.h5"  # or .pth for PyTorch
    
    print(f"🔧 Initializing Persian OCR with model: {model_path}")
    ocr = PersianOCR(model_path)
    
    # Path to your input image containing Persian text
    image_path = "path/to/your/persian_text_image.jpg"
    print(f"📸 Input image: {image_path}")
    
    try:
        # Recognize text in the image
        recognized_text = ocr.recognize_text(image_path)
        print(f"\n🎉 SUCCESS! Recognized text: {recognized_text}")
        
        # Display results
        print("\n📊 Displaying results...")
        ocr.display_results(image_path, recognized_text)
        
    except Exception as e:
        print(f"❌ Error during OCR processing: {e}")
        import traceback
        traceback.print_exc()

# Example usage without a trained model (for testing preprocessing)
def test_preprocessing():
    """
    Test the preprocessing functions without a trained model
    """
    print("🧪 Testing Preprocessing Functions")
    print("=" * 40)
    
    ocr = PersianOCR()  # No model loaded
    print("✅ Initialized OCR without model for testing")
    
    # Test image preprocessing
    image_path = "path/to/your/test_image.jpg"
    print(f"📁 Test image path: {image_path}")
    
    try:
        # Segment characters
        print("\n🔄 Starting character segmentation...")
        characters = ocr.segment_text(image_path)
        print(f"✅ Found {len(characters)} character segments")
        
        # Display segmented characters
        print("📊 Displaying segmented characters...")
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
            print(f"📝 Character {i+1} shape: {char_img.shape}")
        
        plt.tight_layout()
        plt.show()
        print("✅ Preprocessing test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during preprocessing test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🌟 Persian OCR Application Started")
    print("🔤 Ready to recognize Persian text in images!")
    print("=" * 60)
    
    # Run the main OCR function
    print("\n🚀 Option 1: Full OCR Recognition")
    # main()
    
    # Or test preprocessing only
    print("\n🧪 Option 2: Test Preprocessing Only")
    # test_preprocessing()
    
    print("\n💡 To run the application:")
    print("   1. Uncomment 'main()' to run full OCR")
    print("   2. Uncomment 'test_preprocessing()' to test segmentation")
    print("   3. Update the file paths in the functions above")
    print("\n🎯 Happy Persian OCR processing! 🎯")
