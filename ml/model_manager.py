import numpy as np
from typing import Optional, Tuple, Dict
import os

from ml.knn_trainer import KNNTrainer 
from ml.feature_extractor import FeatureExtractor
from utils.database import DatabaseManager
from config import MIN_TRAINING_SAMPLES, MODELS_DIR

class ModelManager:
    """Менеджер для простой и надежной системы"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.feature_extractor = FeatureExtractor()
        self.models_cache = {}
    
    def train_user_model(self, user_id: int) -> Tuple[bool, float, str]:
        """Обучение модели"""
        
        # Получение образцов пользователя
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        if len(user_samples) < MIN_TRAINING_SAMPLES:
            return False, 0.0, f"Недостаточно образцов. Необходимо минимум {MIN_TRAINING_SAMPLES}, собрано {len(user_samples)}"
        
        # Создаем и обучаем модель
        trainer = KNNTrainer(user_id)
        success, accuracy, message = trainer.train_user_model(user_samples)
        
        if success:
            # Кэшируем модель
            self.models_cache[user_id] = trainer
            
            # Обновляем пользователя
            user = self.db.get_user_by_id(user_id)
            if user:
                user.is_trained = True
                user.training_samples = len(user_samples)
                self.db.update_user(user)
            
        
        return success, accuracy, message
    
    def authenticate_user(self, user_id: int, keystroke_features: dict, verbose: bool = False) -> Tuple[bool, float, str]:
        """Аутентификация пользователя"""
       
        
        # Получение модели
        model = self._get_user_model(user_id)
        if model is None:
            
            return False, 0.0, "Модель пользователя не найдена"
        
        # Подготовка вектора признаков
        feature_vector = np.array([
            keystroke_features.get('avg_dwell_time', 0),
            keystroke_features.get('std_dwell_time', 0),
            keystroke_features.get('avg_flight_time', 0),
            keystroke_features.get('std_flight_time', 0),
            keystroke_features.get('typing_speed', 0),
            keystroke_features.get('total_typing_time', 0)
        ])
        
        
        
        # Аутентификация
        is_authenticated, confidence = model.predict(feature_vector)
        
        if is_authenticated:
            message = f"Аутентификация успешна (уверенность: {confidence:.2%})"
        else:
            message = f"Аутентификация отклонена (уверенность: {confidence:.2%})"
        
        
        
        return is_authenticated, confidence, message
    
    def authenticate_user_detailed(self, user_id: int, keystroke_features: dict) -> Tuple[bool, float, dict]:
        """Аутентификация с детальной статистикой"""
        is_authenticated, confidence, message = self.authenticate_user(user_id, keystroke_features, verbose=True)
        
        # Детальная статистика
        detailed_stats = {
            'final_confidence': confidence,
            'threshold': 0.70,
            'training_samples': len(self.db.get_user_keystroke_samples(user_id, training_only=True)),
            'model_type': 'knn'
        }
        
        return is_authenticated, confidence, detailed_stats
    
    def _get_user_model(self, user_id: int) -> Optional[KNNTrainer]:
        """Получение модели пользователя"""
        # Проверка кэша
        if user_id in self.models_cache:
            return self.models_cache[user_id]
        
        # Загрузка модели
        model = KNNTrainer.load_model(user_id)
        if model:
            self.models_cache[user_id] = model
            return model
        
        return None
    
    def delete_user_model(self, user_id: int):
        """Удаление модели"""
        # Удаление из кэша
        if user_id in self.models_cache:
            del self.models_cache[user_id]
        
        # Удаление файла
        model_path = os.path.join(MODELS_DIR, f"user_{user_id}_knn.pkl")
        if os.path.exists(model_path):
            os.remove(model_path)
    
    def get_model_info(self, user_id: int) -> dict:
        """Информация о модели"""
        model = self._get_user_model(user_id)
        user_samples = self.db.get_user_keystroke_samples(user_id, training_only=True)
        
        info = {
            'min_samples': MIN_TRAINING_SAMPLES,
            'training_samples': len(user_samples),
            'model_type': 'none'
        }
        
        if model:
            model_info = model.get_model_info()
            info.update({
                'is_trained': True,
                'model_type': 'knn',
                'best_params': model_info.get('best_params', {}),
                'training_stats': model_info.get('training_stats', {}),
                'feature_importance': []
            })
        else:
            info.update({
                'is_trained': False,
                'feature_importance': []
            })
        
        return info
    
    def get_training_report(self, user_id: int) -> Optional[Dict]:
        """Отчет об обучении"""
        model = self._get_user_model(user_id)
        if model and hasattr(model, 'training_stats'):
            return model.training_stats
        return None