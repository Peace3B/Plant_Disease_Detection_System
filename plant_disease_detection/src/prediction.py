"""
Prediction module for plant disease detection
"""
import numpy as np
from typing import Dict, Union, List
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras

from preprocessing import ImagePreprocessor


class DiseasePredictor:
    """Handles disease prediction for plant images"""
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            metadata_path: Path to model metadata (optional)
        """
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path) if metadata_path else self.model_path.parent / 'model_metadata.json'
        
        # Load model
        self.model = self.load_model()
        
        # Load metadata
        self.metadata = self.load_metadata()
        self.class_names = self.metadata.get('classes', [])
        self.image_size = tuple(self.metadata.get('image_size', (224, 224)))
        
        # Initialize preprocessor
        self.preprocessor = ImagePreprocessor(image_size=self.image_size)
        
        print(f"Predictor initialized with {len(self.class_names)} classes")
    
    def load_model(self) -> keras.Model:
        """Load trained model"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        model = keras.models.load_model(str(self.model_path))
        print(f"Model loaded from: {self.model_path}")
        return model
    
    def load_metadata(self) -> Dict:
        """Load model metadata"""
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata
        else:
            print(f"Warning: Metadata not found at {self.metadata_path}")
            return {}
    
    def predict(self, image: Union[str, Path, bytes]) -> Dict:
        """
        Predict disease for a single image
        
        Args:
            image: Image as file path, bytes, or PIL Image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        processed_image = self.preprocessor.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        
        # Get results
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        predicted_class = self.class_names[predicted_class_idx] if self.class_names else f"Class_{predicted_class_idx}"
        
        # Get all class probabilities
        all_probabilities = {}
        for idx, prob in enumerate(predictions[0]):
            class_name = self.class_names[idx] if self.class_names else f"Class_{idx}"
            all_probabilities[class_name] = float(prob)
        
        # Sort probabilities
        sorted_probs = dict(sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True))
        
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': sorted_probs,
            'top_3_predictions': dict(list(sorted_probs.items())[:3])
        }
        
        return result
    
    def predict_batch(self, images: List[Union[str, Path, bytes]]) -> List[Dict]:
        """
        Predict disease for multiple images
        
        Args:
            images: List of images
            
        Returns:
            List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        
        return results
    
    def predict_with_visualization(self, image: Union[str, Path, bytes]) -> Dict:
        """
        Predict and include visualization data
        
        Args:
            image: Input image
            
        Returns:
            Prediction results with visualization data
        """
        result = self.predict(image)
        
        # Add additional visualization info
        result['visualization'] = {
            'confidence_threshold': 0.7,
            'high_confidence': result['confidence'] > 0.7,
            'prediction_quality': self._assess_prediction_quality(result['confidence'])
        }
        
        return result
    
    def _assess_prediction_quality(self, confidence: float) -> str:
        """Assess the quality of prediction based on confidence"""
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.7:
            return "Good"
        elif confidence >= 0.5:
            return "Fair"
        else:
            return "Poor"
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        info = {
            'model_path': str(self.model_path),
            'num_classes': len(self.class_names),
            'classes': self.class_names,
            'image_size': self.image_size,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_parameters': self.model.count_params(),
            'metadata': self.metadata
        }
        return info
    
    def validate_and_predict(self, image: Union[str, Path, bytes]) -> Dict:
        """
        Validate image before prediction
        
        Args:
            image: Input image
            
        Returns:
            Prediction results or error message
        """
        # Validate image
        if not self.preprocessor.validate_image(image):
            return {
                'error': 'Invalid image format or size',
                'predicted_class': None,
                'confidence': 0.0
            }
        
        # Make prediction
        try:
            result = self.predict(image)
            result['error'] = None
            return result
        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': None,
                'confidence': 0.0
            }


class BatchPredictor:
    """Handles batch predictions efficiently"""
    
    def __init__(self, predictor: DiseasePredictor, batch_size: int = 32):
        self.predictor = predictor
        self.batch_size = batch_size
    
    def predict_directory(self, directory: Union[str, Path]) -> List[Dict]:
        """
        Predict all images in a directory
        
        Args:
            directory: Directory containing images
            
        Returns:
            List of prediction results
        """
        directory = Path(directory)
        image_files = list(directory.glob('*.jpg')) + list(directory.glob('*.png'))
        
        print(f"Found {len(image_files)} images")
        
        results = []
        for img_path in image_files:
            result = self.predictor.predict(str(img_path))
            result['filename'] = img_path.name
            results.append(result)
        
        return results
    
    def predict_with_statistics(self, images: List) -> Dict:
        """
        Predict batch and return statistics
        
        Args:
            images: List of images
            
        Returns:
            Dictionary with predictions and statistics
        """
        results = self.predictor.predict_batch(images)
        
        # Calculate statistics
        confidences = [r['confidence'] for r in results]
        predictions = [r['predicted_class'] for r in results]
        
        stats = {
            'total_predictions': len(results),
            'average_confidence': np.mean(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
            'class_distribution': {},
            'predictions': results
        }
        
        # Count predictions per class
        for pred in predictions:
            stats['class_distribution'][pred] = stats['class_distribution'].get(pred, 0) + 1
        
        return stats


# Helper functions
def predict_single_image(model_path: str, image_path: str) -> Dict:
    """
    Convenience function to predict a single image
    
    Args:
        model_path: Path to model
        image_path: Path to image
        
    Returns:
        Prediction result
    """
    predictor = DiseasePredictor(model_path)
    result = predictor.predict(image_path)
    return result


def format_prediction_output(result: Dict) -> str:
    """
    Format prediction result for display
    
    Args:
        result: Prediction result dictionary
        
    Returns:
        Formatted string
    """
    output = f"""
Prediction Results
==================
Predicted Class: {result['predicted_class']}
Confidence: {result['confidence']:.2%}

Top 3 Predictions:
"""
    for cls, prob in result['top_3_predictions'].items():
        output += f"  {cls}: {prob:.2%}\n"
    
    return output


# Example usage
if __name__ == "__main__":
    # Example prediction
    MODEL_PATH = "models/plant_disease_model.h5"
    
    # Check if model exists
    if Path(MODEL_PATH).exists():
        # Initialize predictor
        predictor = DiseasePredictor(MODEL_PATH)
        
        # Get model info
        info = predictor.get_model_info()
        print("\nModel Info:")
        print(f"Classes: {info['num_classes']}")
        print(f"Parameters: {info['total_parameters']:,}")
        
        # Example prediction (uncomment when you have an image)
        # result = predictor.predict("path/to/test/image.jpg")
        # print(format_prediction_output(result))
    else:
        print(f"Model not found at {MODEL_PATH}")
        print("Please train the model first using model.py")