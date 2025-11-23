"""
Model creation, training, and retraining module
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict
import shutil

from preprocessing import DataGenerator


class PlantDiseaseModel:
    """Plant disease classification model"""
    
    def __init__(
        self,
        num_classes: int,
        image_size: Tuple[int, int] = (224, 224),
        model_path: Optional[str] = None
    ):
        self.num_classes = num_classes
        self.image_size = image_size
        self.model_path = model_path
        self.model = None
        self.history = None
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.model = self.build_model()
    
    def build_model(self) -> keras.Model:
        """Build CNN model architecture"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.image_size, 3)),
            
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def compile_model(self, learning_rate: float = 0.001):
        """Compile the model with optimizer and loss"""
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
    
    def train(
        self,
        train_dir: str,
        epochs: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
        callbacks: Optional[list] = None
    ) -> keras.callbacks.History:
        """
        Train the model
        
        Args:
            train_dir: Directory containing training data
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
            callbacks: List of callbacks
            
        Returns:
            Training history
        """
        # Create data generators
        train_gen, val_gen = DataGenerator.create_generators(
            train_dir=train_dir,
            batch_size=batch_size,
            image_size=self.image_size,
            validation_split=validation_split
        )
        
        # Default callbacks if none provided
        if callbacks is None:
            callbacks = self.get_default_callbacks()
        
        # Compile model if not already compiled
        if not self.model.optimizer:
            self.compile_model()
        
        # Train
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def get_default_callbacks(self, model_save_path: str = "models/best_model.h5") -> list:
        """Get default training callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def evaluate(self, test_dir: str, batch_size: int = 32) -> Dict:
        """
        Evaluate model on test data
        
        Args:
            test_dir: Directory containing test data
            batch_size: Batch size
            
        Returns:
            Dictionary with evaluation metrics
        """
        test_gen = DataGenerator.create_test_generator(
            test_dir=test_dir,
            batch_size=batch_size,
            image_size=self.image_size
        )
        
        results = self.model.evaluate(test_gen, verbose=1)
        
        metrics = {
            'loss': float(results[0]),
            'accuracy': float(results[1]),
            'precision': float(results[2]),
            'recall': float(results[3])
        }
        
        # Calculate F1 score
        precision = metrics['precision']
        recall = metrics['recall']
        metrics['f1_score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return metrics
    
    def save_model(self, save_path: str, metadata: Optional[Dict] = None):
        """
        Save model and metadata
        
        Args:
            save_path: Path to save model
            metadata: Additional metadata to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(str(save_path))
        print(f"Model saved to: {save_path}")
        
        # Save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'saved_at': datetime.now().isoformat(),
            'image_size': self.image_size,
            'num_classes': self.num_classes,
            'model_path': str(save_path)
        })
        
        metadata_path = save_path.parent / 'model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def load_model(self, model_path: str):
        """Load model from file"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    def retrain(
        self,
        new_data_dir: str,
        epochs: int = 10,
        batch_size: int = 32,
        fine_tune: bool = True
    ) -> keras.callbacks.History:
        """
        Retrain model with new data
        
        Args:
            new_data_dir: Directory with new training data
            epochs: Number of epochs
            batch_size: Batch size
            fine_tune: If True, use fine-tuning (lower learning rate)
            
        Returns:
            Training history
        """
        # Backup current model
        backup_path = Path("models/backup")
        backup_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"model_backup_{timestamp}.h5"
        self.model.save(str(backup_file))
        print(f"Model backed up to: {backup_file}")
        
        # Set learning rate for fine-tuning
        learning_rate = 0.0001 if fine_tune else 0.001
        self.compile_model(learning_rate=learning_rate)
        
        # Train on new data
        history = self.train(
            train_dir=new_data_dir,
            epochs=epochs,
            batch_size=batch_size
        )
        
        print("Retraining completed!")
        return history
    
    def get_model_summary(self) -> str:
        """Get model architecture summary"""
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary


def train_new_model(
    train_dir: str,
    num_classes: int,
    class_names: list,
    epochs: int = 20,
    save_path: str = "models/plant_disease_model.h5"
):
    """
    Train a new model from scratch
    
    Args:
        train_dir: Training data directory
        num_classes: Number of classes
        class_names: List of class names
        epochs: Number of epochs
        save_path: Path to save model
    """
    print("=== Training New Model ===")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # Create model
    model = PlantDiseaseModel(num_classes=num_classes)
    
    # Train
    history = model.train(
        train_dir=train_dir,
        epochs=epochs,
        batch_size=32
    )
    
    # Save
    metadata = {
        'classes': class_names,
        'num_classes': num_classes,
        'epochs': epochs,
        'architecture': 'Custom CNN'
    }
    model.save_model(save_path, metadata=metadata)
    
    print("\n=== Training Complete ===")
    return model, history


# Example usage
if __name__ == "__main__":
    # Example configuration
    TRAIN_DIR = "data/train"
    NUM_CLASSES = 6
    CLASS_NAMES = [
        'Tomato_Bacterial_spot',
        'Tomato_Early_blight',
        'Tomato_Late_blight',
        'Tomato_Leaf_Mold',
        'Tomato_Septoria_leaf_spot',
        'Tomato_healthy'
    ]
    
    # Train new model
    # model, history = train_new_model(
    #     train_dir=TRAIN_DIR,
    #     num_classes=NUM_CLASSES,
    #     class_names=CLASS_NAMES,
    #     epochs=20
    # )
    
    print("Model module ready!")