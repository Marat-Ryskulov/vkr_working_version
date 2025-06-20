import os

# Основные настройки
APP_NAME = "Двухфакторная аутентификация"

# Пути к файлам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
DATABASE_PATH = os.path.join(DATA_DIR, "users.db")

# Создание директорий
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Фиксированные размеры окон
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
TRAINING_WINDOW_WIDTH = 700
TRAINING_WINDOW_HEIGHT = 600
STATS_WINDOW_WIDTH = 1000
STATS_WINDOW_HEIGHT = 700
FONT_SIZE = 10
FONT_FAMILY = "Arial"

MIN_TRAINING_SAMPLES = 50

# Настройки безопасности
SALT_LENGTH = 32

# Панграмма для обучения и аутентификации
PANGRAM = "The quick brown fox jumps over the lazy dog"

# Путь для дополнительных данных
CSV_EXPORTS_DIR = os.path.join(DATA_DIR, "csv_exports")

# Создание дополнительной директории
os.makedirs(CSV_EXPORTS_DIR, exist_ok=True)