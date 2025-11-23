"""
Data preprocessing module for plant disease detection
"""
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Union, Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class ImagePreprocessor:
    """Handles image preprocessing operations"""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224)):
        self.image_size = image_size
    
    def preprocess_image(self, image: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
        """
        Preprocess a single image for model prediction
        
        Args:
            image: Image as file path, bytes, or PIL Image
            
        Returns:
            Preprocessed image array ready for model input
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize
        img = img.resize(self.image_size, Image.Resampling.LANCZOS)
        
        # Convert to array
        img_array = np.array(img, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def preprocess_batch(self, images: list) -> np.ndarray:
        """
        Preprocess multiple images
        
        Args:
            images: List of images (paths, bytes, or PIL Images)
            
        Returns:
            Batch of preprocessed images
        """
        preprocessed = []
        for img in images:
            processed = self.preprocess_image(img)
            preprocessed.append(processed[0])  # Remove batch dimension
        
        return np.array(preprocessed)
    
    def validate_image(self, image: Union[str, Path, bytes]) -> bool:
        """
        Validate if image is suitable for processing
        
        Args:
            image: Image to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            else:
                return False
            
            # Check if image can be converted to RGB
            img.convert('RGB')
            
            # Check minimum size
            if img.size[0] < 50 or img.size[1] < 50:
                return False
            
            return True
        except Exception:
            return False
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to an image
        
        Args:
            image: Input image array
            
        Returns:
            Augmented image
        """
        # Random flip
        if np.random.random() > 0.5:
            image = np.fliplr(image)
        
        # Random rotation
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        
        # Random brightness adjustment
        factor = np.random.uniform(0.8, 1.2)
        image = np.clip(image * factor, 0, 1)
        
        return image


class DataGenerator:
    """Creates data generators for training and validation"""
    
    @staticmethod
    def create_generators(
        train_dir: Union[str, Path],
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224),
        validation_split: float = 0.2
    ) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """
        Create training and validation data generators
        
        Args:
            train_dir: Directory containing training data
            batch_size: Batch size for training
            image_size: Target image size
            validation_split: Fraction of data to use for validation
            
        Returns:
            Tuple of (train_generator, validation_generator)
        """
        # Training data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator (same augmentation for consistency)
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    @staticmethod
    def create_test_generator(
        test_dir: Union[str, Path],
        batch_size: int = 32,
        image_size: Tuple[int, int] = (224, 224)
    ) -> ImageDataGenerator:
        """
        Create test data generator (no augmentation)
        
        Args:
            test_dir: Directory containing test data
            batch_size: Batch size for testing
            image_size: Target image size
            
        Returns:
            Test data generator
        """
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False
        )
        
        return test_generator


def load_and_preprocess_dataset(data_dir: Union[str, Path]) -> dict:
    """
    Load and analyze dataset structure
    
    Args:
        data_dir: Root directory of dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)
    
    stats = {
        'classes': [],
        'class_counts': {},
        'total_images': 0
    }
    
    # Get all subdirectories (classes)
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    stats['classes'] = classes
    
    # Count images per class
    for cls in classes:
        cls_path = data_dir / cls
        images = list(cls_path.glob('*.jpg')) + list(cls_path.glob('*.png'))
        count = len(images)
        stats['class_counts'][cls] = count
        stats['total_images'] += count
    
    return stats


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(image_size=(224, 224))
    
    # Test single image preprocessing
    print("Image Preprocessor initialized")
    print(f"Target image size: {preprocessor.image_size}")
    
    # Test data generator
    print("\nTesting data generator...")
    # Uncomment and modify path as needed
    # train_gen, val_gen = DataGenerator.create_generators(
    #     train_dir="data/train",
    #     batch_size=32
    # )
    # print(f"Training samples: {train_gen.samples}")
    # print(f"Validation samples: {val_gen.samples}")