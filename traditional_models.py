"""Traditional ML models for music genre classification."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import joblib
from typing import Dict, Tuple, Any


class TraditionalModelTrainer:
    """Trainer for traditional ML models."""
    
    def __init__(self, model_type: str, config: Dict = None):
        """
        Args:
            model_type: Type of model ('random_forest', 'svm', 'knn', 'gradient_boosting')
            config: Configuration dictionary for hyperparameters
        """
        self.model_type = model_type
        self.config = config or {}
        self.model = self._create_model()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def _create_model(self) -> Any:
        """Create the specified model instance."""
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 20),
                min_samples_split=self.config.get('min_samples_split', 5),
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                C=self.config.get('C', 10),
                gamma=self.config.get('gamma', 'scale'),
                kernel=self.config.get('kernel', 'rbf'),
                random_state=42,
                probability=True
            )
        elif self.model_type == 'knn':
            return KNeighborsClassifier(
                n_neighbors=self.config.get('n_neighbors', 11),
                weights=self.config.get('weights', 'distance'),
                metric=self.config.get('metric', 'euclidean')
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                learning_rate=self.config.get('learning_rate', 0.1),
                max_depth=self.config.get('max_depth', 5),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training (scale and encode labels).
        
        Args:
            X: Feature matrix
            y: Labels
        
        Returns:
            Tuple of (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded)
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the model."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions (if supported)."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        return None
    
    def save_model(self, path: str) -> None:
        """Save model, scaler, and label encoder."""
        joblib.dump(self.model, f"{path}/{self.model_type}_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoder, f"{path}/label_encoder.pkl")
    
    def load_model(self, path: str) -> None:
        """Load model, scaler, and label encoder."""
        self.model = joblib.load(f"{path}/{self.model_type}_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoder = joblib.load(f"{path}/label_encoder.pkl")


def get_hyperparameter_grid(model_type: str) -> Dict:
    """
    Get hyperparameter grid for tuning.
    
    Args:
        model_type: Type of model
    
    Returns:
        Dictionary of hyperparameter grids
    """
    grids = {
        'random_forest': {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'svm': {
            'C': [0.1, 1, 10, 50, 100, 200],
            'gamma': ['scale', 'auto', 0.01, 0.001, 0.0001],
            'kernel': ['rbf', 'poly', 'sigmoid'],
            'degree': [2, 3, 4],
            'coef0': [0.0, 0.1, 0.5, 1.0],
            'shrinking': [True, False],
            'class_weight': ['balanced', None]
        },
        'knn': {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev'],
            'p': [1, 2, 3],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [20, 30, 40, 50]
        },
        'gradient_boosting': {
            'n_estimators': [100, 150, 200, 300, 400],
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 7],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2', None]
        }
    }
    return grids.get(model_type, {})