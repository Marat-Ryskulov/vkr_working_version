import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from datetime import datetime
import json

from models.user import User
from auth.keystroke_auth import KeystrokeAuthenticator
from ml.model_manager import ModelManager
from utils.database import DatabaseManager
from config import FONT_FAMILY

plt.style.use('default')

class StatsWindow:
    """Статистика с отдельными вкладками для каждого признака"""
    
    def __init__(self, parent, user: User, keystroke_auth: KeystrokeAuthenticator):
        self.parent = parent
        self.user = user
        self.keystroke_auth = keystroke_auth
        self.model_manager = ModelManager()
        self.db = DatabaseManager()
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Статистика клавиатурного почерка - {user.username}")
        self.window.geometry("1000x700")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Получение данных
        self.training_samples = self.db.get_user_training_samples(user.id)
        
        # Создание интерфейса
        self.create_interface()
        self.load_statistics()
    
    def create_interface(self):
        """Создание интерфейса с прокруткой"""
        # Создаем главный контейнер с прокруткой
        main_canvas = tk.Canvas(self.window)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Привязка колесика мыши для прокрутки
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Заголовок
        header_frame = ttk.Frame(scrollable_frame, padding=10)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text=f"Статистика клавиатурного почерка - {self.user.username}",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack()
        
        # Основная информация
        info_frame = ttk.LabelFrame(header_frame, text="Информация", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.info_text = tk.Text(info_frame, height=4, width=100, font=(FONT_FAMILY, 10))
        self.info_text.pack()
        
        # ✅ ИЗМЕНЕНИЕ: Создаем Notebook с 4 отдельными вкладками для признаков
        features_frame = ttk.LabelFrame(scrollable_frame, text="Анализ признаков клавиатурного почерка", padding=15)
        features_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.features_notebook = ttk.Notebook(features_frame)
        self.features_notebook.pack(fill=tk.X, pady=10)
        
        # Устанавливаем фиксированную высоту для notebook
        self.features_notebook.configure(height=500)
        
        # Вкладка 1: Время удержания клавиш
        tab1 = ttk.Frame(self.features_notebook)
        self.features_notebook.add(tab1, text="Время удержания")
        self.create_feature_tab(tab1, 0, "Время удержания клавиш (мс)", 'skyblue')
        
        # Вкладка 2: Время между клавишами
        tab2 = ttk.Frame(self.features_notebook)
        self.features_notebook.add(tab2, text="Время между клавишами")
        self.create_feature_tab(tab2, 2, "Время между клавишами (мс)", 'lightcoral')
        
        # Вкладка 3: Скорость печати
        tab3 = ttk.Frame(self.features_notebook)
        self.features_notebook.add(tab3, text="Скорость печати")
        self.create_feature_tab(tab3, 4, "Скорость печати (клавиш/сек)", 'lightgreen')
        
        # Вкладка 4: Общее время ввода
        tab4 = ttk.Frame(self.features_notebook)
        self.features_notebook.add(tab4, text="Общее время")
        self.create_feature_tab(tab4, 5, "Общее время ввода (сек)", 'lightsalmon')
        
        # Кнопки
        self.create_buttons(scrollable_frame)
    
    def create_feature_tab(self, parent_frame, feature_index, feature_name, color):
        """Создание упрощенной вкладки для отдельного признака"""
        # Фрейм для графика
        plot_frame = ttk.Frame(parent_frame, padding=10)
        plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # Сохраняем ссылки для заполнения данными
        setattr(self, f'plot_frame_{feature_index}', plot_frame)
        setattr(self, f'feature_name_{feature_index}', feature_name)
        setattr(self, f'color_{feature_index}', color)
    
    def create_buttons(self, parent_frame):
        """Создание кнопок"""
        buttons_frame = ttk.Frame(parent_frame)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            buttons_frame,
            text="Экспорт данных",
            command=self.export_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Обновить",
            command=self.refresh_data
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            buttons_frame,
            text="Закрыть",
            command=self.window.destroy
        ).pack(side=tk.RIGHT, padx=5)
    
    def load_statistics(self):
        """Загрузка статистики"""
        try:
            self.load_general_info()
            self.load_features_analysis()
            
        except Exception as e:
            print(f"Ошибка загрузки статистики: {e}")
            import traceback
            traceback.print_exc()
    
    def load_general_info(self):
        """Загрузка общей информации"""
        n_samples = len(self.training_samples)
        
        if n_samples == 0:
            info = "Нет данных для анализа"
            self.info_text.insert(tk.END, info)
            return
        
        # Извлечение признаков для анализа
        features_data = []
        for sample in self.training_samples:
            if sample.features:
                features_data.append([
                    sample.features.get('avg_dwell_time', 0),
                    sample.features.get('avg_flight_time', 0),
                    sample.features.get('typing_speed', 0),
                    sample.features.get('total_typing_time', 0)
                ])
        
        if not features_data:
            info = "Признаки не рассчитаны для образцов"
            self.info_text.insert(tk.END, info)
            return
        
        features_array = np.array(features_data)
        
        # Статистика
        info = f"""Пользователь: {self.user.username}
Статус модели: {'Обучена' if self.user.is_trained else 'Не обучена'}
Количество образцов: {n_samples}

Характеристики клавиатурного почерка:
Время удержания клавиш: {np.mean(features_array[:, 0])*1000:.1f} ± {np.std(features_array[:, 0])*1000:.1f} мс
Время между клавишами: {np.mean(features_array[:, 1])*1000:.1f} ± {np.std(features_array[:, 1])*1000:.1f} мс  
Скорость печати: {np.mean(features_array[:, 2]):.1f} ± {np.std(features_array[:, 2]):.1f} клавиш/сек
Общее время ввода: {np.mean(features_array[:, 3]):.1f} ± {np.std(features_array[:, 3]):.1f} сек"""
        
        self.info_text.insert(tk.END, info)
    
    def load_features_analysis(self):
        """Анализ каждого признака отдельно"""
        if not self.training_samples:
            return
        
        try:
            # Извлечение данных признаков
            features_data = []
            for sample in self.training_samples:
                if sample.features:
                    features_data.append([
                        sample.features.get('avg_dwell_time', 0) * 1000,  # в мс
                        sample.features.get('avg_flight_time', 0) * 1000,  # в мс
                        sample.features.get('typing_speed', 0),
                        sample.features.get('total_typing_time', 0)
                    ])
            
            if not features_data:
                return
            
            features_array = np.array(features_data)
            
            # Индексы признаков для обработки
            feature_indices = [0, 2, 4, 5]  # dwell_time, flight_time, speed, total_time
            
            for i, feature_idx in enumerate(feature_indices):
                self.create_individual_analysis(features_array[:, i], feature_idx)
                
        except Exception as e:
            print(f"Ошибка анализа признаков: {e}")
    
    def create_individual_analysis(self, data, feature_index):
        """Создание упрощенного анализа для признака - только гистограмма с средним"""
        try:
            # Получаем компоненты для этого признака
            plot_frame = getattr(self, f'plot_frame_{feature_index}')
            feature_name = getattr(self, f'feature_name_{feature_index}')
            color = getattr(self, f'color_{feature_index}')
            
            # Вычисляем только среднее значение
            mean_val = np.mean(data)
            
            # ✅ УПРОЩЕННАЯ ГИСТОГРАММА - только среднее значение
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            
            # Гистограмма
            n_bins = min(15, len(data)//2 + 1)
            ax.hist(data, bins=n_bins, alpha=0.7, 
                   color=color, edgecolor='black', linewidth=1)
            
            # Только линия среднего значения
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=3, 
                      label=f'Среднее: {mean_val:.2f}')
            
            # Простое оформление
            ax.set_xlabel(feature_name, fontsize=11)
            ax.set_ylabel('Частота', fontsize=11)
            ax.set_title(f'Распределение: {feature_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            fig.tight_layout()
            
            # Встраиваем в интерфейс
            canvas = FigureCanvasTkAgg(fig, plot_frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            canvas.draw()
            
        except Exception as e:
            print(f"Ошибка создания анализа для признака {feature_index}: {e}")

    
    def refresh_data(self):
        """Обновление данных"""
        try:
            # Перезагружаем данные
            self.training_samples = self.db.get_user_training_samples(self.user.id)
            
            # Очищаем текстовое поле общей информации
            self.info_text.delete('1.0', tk.END)
            
            # Перезагружаем статистику
            self.load_statistics()
            
            messagebox.showinfo("Обновление", "Данные обновлены")
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обновления: {e}")
    
    def export_data(self):
        """Экспорт данных в файл"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON файлы", "*.json"), ("CSV файлы", "*.csv"), ("Все файлы", "*.*")],
                title="Экспорт данных"
            )
            
            if filename:
                if filename.endswith('.csv'):
                    self.export_to_csv(filename)
                else:
                    self.export_to_json(filename)
                
                messagebox.showinfo("Экспорт", f"Данные экспортированы: {filename}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта: {e}")
    
    def export_to_json(self, filename: str):
        """Экспорт в JSON"""
        data = {
            'user': self.user.username,
            'export_date': datetime.now().isoformat(),
            'total_samples': len(self.training_samples),
            'samples': []
        }
        
        for i, sample in enumerate(self.training_samples):
            sample_data = {
                'sample_id': i + 1,
                'timestamp': sample.timestamp.isoformat(),
                'features': sample.features
            }
            data['samples'].append(sample_data)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def export_to_csv(self, filename: str):
        """Экспорт в CSV"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'sample_id', 'timestamp', 
                'avg_dwell_time', 'std_dwell_time',
                'avg_flight_time', 'std_flight_time',
                'typing_speed', 'total_typing_time'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for i, sample in enumerate(self.training_samples):
                row = {
                    'sample_id': i + 1,
                    'timestamp': sample.timestamp.isoformat(),
                    **sample.features
                }
                writer.writerow(row)