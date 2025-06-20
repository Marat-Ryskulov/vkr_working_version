
"""
Система двухфакторной аутентификации с динамикой нажатий клавиш
Главный файл запуска приложения
"""

import sys
import os
from pathlib import Path

# Добавляем корневую директорию в путь Python
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

# Импортируем главное окно
from gui.main_window import MainWindow

def main():
    """Точка входа в приложение"""
    
    app = MainWindow()
    app.run()
        

if __name__ == "__main__":
    main()