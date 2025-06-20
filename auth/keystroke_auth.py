from typing import Tuple, Dict
from datetime import datetime

from models.user import User
from models.keystroke_data import KeystrokeData
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from utils.security import SecurityManager

class KeystrokeAuthenticator:
    """Класс для аутентификации по динамике нажатий клавиш"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.model_manager = ModelManager()
        self.security = SecurityManager()
        self.current_session = {}  # Текущие сессии записи нажатий
    
    def start_keystroke_recording(self, user_id: int) -> str:
        """
        Начало записи динамики нажатий
        Возвращает session_id
        """
        session_id = self.security.generate_session_id()
        
        self.current_session[session_id] = KeystrokeData(
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now()
        )
        
        return session_id
    
    def record_key_event(self, session_id: str, key: str, event_type: str):
        """Запись события клавиши"""
        if session_id not in self.current_session:
            raise ValueError("Сессия не найдена")
        
        self.current_session[session_id].add_key_event(key, event_type)
    
    def finish_recording(self, session_id: str, is_training: bool = False) -> Dict[str, float]:
        """
        Завершение записи и извлечение признаков
        Возвращает словарь признаков
        """
        if session_id not in self.current_session:
            raise ValueError("Сессия не найдена")
    
        keystroke_data = self.current_session[session_id]
    
        #Вычисляем признаки
        features = keystroke_data.calculate_features()
    
        # Проверяем, что признаки были рассчитаны
        if not features or all(v == 0 for v in features.values()):
            # Создаем пустые признаки для совместимости
            features = {
                'avg_dwell_time': 0.0,
                'std_dwell_time': 0.0,
                'avg_flight_time': 0.0,
                'std_flight_time': 0.0,
                'typing_speed': 0.0,
                'total_typing_time': 0.0
            }
            keystroke_data.features = features
    
        # Сохранение в БД, если это обучающий образец
        if is_training:
            try:
                self.db.save_keystroke_sample(keystroke_data, is_training=True)
            except Exception as e:
                # Сохранение сырых данных о нажатиях
                user = self.db.get_user_by_id(keystroke_data.user_id)
                if user:
                    try:
                        keystroke_data.save_raw_events_to_csv(user.id, user.username)
                    except Exception as e:
                        print(f"Ошибка сохранения CSV: {e}")
    
        # Удаление из текущих сессий
        del self.current_session[session_id]
    
        return features
    
    def authenticate(self, user, keystroke_features: Dict[str, float]) -> Tuple[bool, float, str]:
        """
        Аутентификация со статистикой
        """
        if not user.is_trained:
            return False, 0.0, "Модель пользователя не обучена."

        # Аутентификация через ModelManager
        is_authenticated, confidence, detailed_stats = self.model_manager.authenticate_user_detailed(
            user.id, keystroke_features
        )

        threshold = detailed_stats.get('threshold', 0.7)

        

        # Формируем сообщение
        if is_authenticated:
            message = f"Аутентификация успешна (уверенность: {confidence:.1%})"
        else:
            message = f"Аутентификация отклонена (уверенность: {confidence:.1%})"

        # Сохранение попытки аутентификации в базу данных
        try:
            auth_session_id = self.security.generate_session_id()
        
            self.db.save_auth_attempt(
                user_id=user.id,
                session_id=auth_session_id,
                features=keystroke_features,
                knn_confidence=confidence,
                distance_score=0.0,  
                feature_score=0.0,   
                final_confidence=confidence,
                threshold=detailed_stats.get('threshold', 0.7),
                result=is_authenticated
            )
        except Exception as e:
            print(f"Ошибка сохранения попытки аутентификации: {e}")

        return is_authenticated, confidence, message
    
    def train_user_model(self, user: User) -> Tuple[bool, float, str]:
        """
        Обучение модели пользователя
        Возвращает: (успех, точность, сообщение)
        """
        return self.model_manager.train_user_model(user.id)
    
    def get_training_progress(self, user: User) -> Dict[str, any]:
        """Получение прогресса обучения пользователя"""
        samples = self.db.get_user_training_samples(user.id)
        
        from config import MIN_TRAINING_SAMPLES
        
        progress = {
            'current_samples': len(samples),
            'required_samples': MIN_TRAINING_SAMPLES,
            'progress_percent': min(100, (len(samples) / MIN_TRAINING_SAMPLES) * 100),
            'is_ready': len(samples) >= MIN_TRAINING_SAMPLES,
            'is_trained': user.is_trained
        }
        
        return progress
    
    def reset_user_model(self, user: User) -> Tuple[bool, str]:
        """Сброс модели пользователя и обучающих данных"""
        try:
            # Удаление модели
            self.model_manager.delete_user_model(user.id)
            
            # Удаление обучающих образцов из БД
            self.db.delete_user_samples(user.id)
            
            # Обновление статуса пользователя
            user.is_trained = False
            user.training_samples = 0
            self.db.update_user(user)
            
            print(f"Модель пользователя {user.username} успешно сброшена")
            return True, "Модель и обучающие данные успешно сброшены"
        except Exception as e:
            return False, f"Ошибка при сбросе модели: {str(e)}"
    
    def get_authentication_stats(self, user: User) -> Dict[str, any]:
        """Получение статистики аутентификации пользователя"""

        # Обучающие образцы
        training_samples = self.db.get_user_training_samples(user.id)
    
        # Все образцы (включая попытки аутентификации)
        all_samples = self.db.get_user_keystroke_samples(user.id, training_only=False)
    
        # Попытки аутентификации из отдельной таблицы
        auth_attempts = self.db.get_auth_attempts(user.id, limit=100)
    
        stats = {
            'total_samples': len(all_samples),
            'training_samples': len(training_samples),
            'authentication_attempts': len(auth_attempts),
            'model_info': self.model_manager.get_model_info(user.id)
        }

        return stats