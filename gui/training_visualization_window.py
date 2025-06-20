import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np 
from typing import Dict
from datetime import datetime
import json

from models.user import User
from config import FONT_FAMILY

class TrainingVisualizationWindow:
    """Окно для отображения результатов обучения модели"""
    
    def __init__(self, parent, user: User, training_results: Dict):
        self.parent = parent
        self.user = user
        self.results = training_results
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title(f"Результаты обучения модели - {user.username}")
        self.window.geometry("1200x800")
        self.window.resizable(True, True)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        self.create_interface()
    
    def create_interface(self):
        """Создание интерфейса"""
        # Заголовок
        header_frame = ttk.Frame(self.window, padding=10)
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(
            header_frame,
            text=f"Результаты обучения модели - {self.user.username}",
            font=(FONT_FAMILY, 16, 'bold')
        )
        title_label.pack()
        
        # Основной контейнер с прокруткой
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
        
        # Привязка колесика мыши
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Текстовые результаты
        text_frame = ttk.LabelFrame(scrollable_frame, text="Отчет об обучении", padding=10)
        text_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.results_text = tk.Text(text_frame, height=12, width=120, font=(FONT_FAMILY, 9))
        text_scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=text_scrollbar.set)
        
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        text_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    
        charts_frame = ttk.LabelFrame(scrollable_frame, text="Визуализация результатов", padding=10)
        charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.charts_notebook = ttk.Notebook(charts_frame)
        self.charts_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка 1: FAR/FRR/EER метрики
        tab1 = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(tab1, text="FAR/FRR/EER")
        self.create_far_frr_eer_tab(tab1)

        # Вкладка 2: ROC-кривая
        tab2 = ttk.Frame(self.charts_notebook)
        self.charts_notebook.add(tab2, text="ROC-кривая")
        self.create_roc_tab(tab2)
        
        # Кнопки
        buttons_frame = ttk.Frame(scrollable_frame, padding=10)
        buttons_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(buttons_frame, text="Сохранить отчет", 
                command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Закрыть", 
                command=self.window.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Генерируем и показываем отчет
        try:
            report = self.generate_report()
            self.results_text.insert('1.0', report)
            self.results_text.config(state=tk.DISABLED)
        except Exception as e:
            error_msg = f"Ошибка генерации отчета: {e}"
            print(error_msg)
            self.results_text.insert('1.0', error_msg)
    
    def generate_report(self) -> str:
        """Генерация текстового отчета"""
        try:
            results = self.results
            
            training_samples = results.get('training_samples', 0)
            total_samples = results.get('total_samples', 0)
            test_accuracy = results.get('test_accuracy', 0)
            cv_accuracy = results.get('cv_accuracy', 0)
            precision = results.get('precision', 0)
            recall = results.get('recall', 0)
            roc_auc = results.get('roc_auc', 0)
            best_params = results.get('best_params', {})
            
            far = results.get('far')
            frr = results.get('frr')
            eer = results.get('eer')
            
            has_biometric_metrics = (far is not None and frr is not None and eer is not None)
            
            report = f"""ОТЧЕТ ОБ ОБУЧЕНИИ МОДЕЛИ

Пользователь: {self.user.username}

ДАННЫЕ ОБУЧЕНИЯ:
• Обучающих образцов: {training_samples}
• Всего образцов (с негативными): {total_samples}


ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:
{self._format_params(best_params)}

МЕТРИКИ КАЧЕСТВА МОДЕЛИ:
• Test Accuracy: {test_accuracy:.1%}
  Доля правильно классифицированных образцов на тестовой выборке
  
• CV Accuracy: {cv_accuracy:.1%}
  Средняя точность по кросс-валидации
  
• Precision: {precision:.1%}
  Доля истинно положительных среди всех положительных предсказаний
  (Насколько точно система определяет легитимного пользователя)
  
• Recall: {recall:.1%}
  Доля найденных истинно положительных от всех истинно положительных
  (Насколько полно система находит легитимного пользователя)
  
• ROC AUC: {roc_auc:.1%}
  Площадь под ROC-кривой (способность разделять классы)"""
            
            if has_biometric_metrics:
                report += f"""

БИОМЕТРИЧЕСКИЕ МЕТРИКИ (при пороге 70%):
• FAR (False Acceptance Rate): {far:.2f}%
  Доля ошибочно принятых имитаторов
  
• FRR (False Rejection Rate): {frr:.2f}%
  Доля ошибочно отклоненных легитимных пользователей
  
• EER (Equal Error Rate): {eer:.2f}%
  Средняя ошибка системы (компромисс между FAR и FRR)
    """
                
            else:
                report += f"""

БИОМЕТРИЧЕСКИЕ МЕТРИКИ:
Метрики FAR/FRR/EER недоступны для этой модели.
Переобучите модель, чтобы получить биометрические метрики."""
            
           
            return report
            
        except Exception as e:
            return f"Ошибка генерации отчета: {str(e)}\n\nДанные: {self.results}"
    
    def _format_params(self, params: Dict) -> str:
        """Форматирование параметров"""
        if not params:
            return "• Параметры не определены"
        
        formatted = []
        for key, value in params.items():
            if key == 'n_neighbors':
                formatted.append(f"• Количество соседей (k): {value}")
            elif key == 'weights':
                formatted.append(f"• Веса соседей: {value}")
            elif key == 'metric':
                formatted.append(f"• Метрика расстояния: {value}")
            
        
        return "\n".join(formatted)
    

    
    
    def create_far_frr_eer_tab(self, parent_frame):
        """Вкладка с графиком FAR/FRR"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Анализ биометрических метрик FAR/FRR', fontsize=14, fontweight='bold')
        
        self._plot_far_frr_vs_threshold(ax)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        canvas.draw()
    
    def _plot_far_frr_vs_threshold(self, ax):
        """График FAR и FRR"""
        all_thresholds_data = self.results.get('all_thresholds_data', [])
        
        if not all_thresholds_data:
            ax.text(0.5, 0.5, 'Данные FAR/FRR недоступны\n\nПереобучите модель для получения\nбиометрических метрик', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
            ax.set_title('FAR и FRR', fontsize=12)
            return
        
        thresholds = [r['threshold'] * 100 for r in all_thresholds_data]
        far_values = [r['far'] for r in all_thresholds_data]
        frr_values = [r['frr'] for r in all_thresholds_data]
        
        ax.plot(thresholds, far_values, 'r-o', label='FAR', linewidth=2, markersize=4)
        ax.plot(thresholds, frr_values, 'b-s', label='FRR', linewidth=2, markersize=4)
        ax.axvline(70, color='gray', linestyle='--', alpha=0.7, label='Текущий порог (70%)')
        
       
        
        ax.set_xlabel('Порог (%)', fontsize=11)
        ax.set_ylabel('Частота ошибок (%)', fontsize=11)
        ax.set_title('FAR и FRR', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    
    def create_roc_tab(self, parent_frame):
        """Вкладка с ROC-кривой"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle('ROC-кривая', fontsize=14, fontweight='bold')
        
        self._plot_roc_curve(ax)
        
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, parent_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
        canvas.draw()
    

    
    def _plot_roc_curve(self, ax):
        """График ROC-кривой"""
        
        fpr = self.results.get('fpr')
        tpr = self.results.get('tpr')
        roc_auc = self.results.get('roc_auc')
        
        if fpr and tpr and roc_auc is not None:
            
            # Конвертируем в numpy массивы
            fpr_array = np.array(fpr)
            tpr_array = np.array(tpr)
            
            ax.plot(fpr_array, tpr_array, color='darkorange', lw=3, 
                   label=f'ROC кривая (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Случайный классификатор')
            
            
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'ROC Кривая (AUC = {roc_auc:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        else:
            y_test = self.results.get('y_test')
            y_proba = self.results.get('y_proba')
            
            if y_test and y_proba:
                try:
                    from sklearn.metrics import roc_curve, auc
                    
                    
                    # Конвертируем в numpy
                    y_test_array = np.array(y_test)
                    y_proba_array = np.array(y_proba)
                    
                    # Проверяем наличие обоих классов
                    unique_classes = np.unique(y_test_array)
                    if len(unique_classes) < 2:
                        ax.text(0.5, 0.5, 'ROC недоступна:\nВ тестовых данных только один класс', 
                               ha='center', va='center', transform=ax.transAxes, fontsize=12)
                        return
                    
                    # Строим ROC-кривую
                    fpr, tpr, thresholds = roc_curve(y_test_array, y_proba_array)
                    roc_auc_calc = auc(fpr, tpr)
                    
                    # График ROC-кривой
                    ax.plot(fpr, tpr, color='darkorange', lw=3, 
                           label=f'ROC кривая (AUC = {roc_auc_calc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Случайный классификатор')
                    
                    
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Кривая (AUC = {roc_auc_calc:.3f})')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    
                except ImportError:
                    ax.text(0.5, 0.5, 'sklearn не доступен\nROC кривая недоступна', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                except Exception as e:
                    ax.text(0.5, 0.5, f'Ошибка построения ROC:\n{str(e)}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
            
            else:
                # Создаем примерную ROC-кривую на основе метрик
                
                # Получаем метрики
                precision = self.results.get('precision', 0.85)
                recall = self.results.get('recall', 0.90)
                
                # Создаем примерную ROC-кривую
                if precision > 0 and recall > 0:
                    # Примерные точки ROC на основе precision и recall
                    fpr_points = [0.0, 1-precision, 0.5, 1.0]
                    tpr_points = [0.0, recall, 0.75, 1.0]
                    
                    estimated_auc = np.trapz(tpr_points, fpr_points)
                    
                    ax.plot(fpr_points, tpr_points, color='darkorange', lw=3, 
                           label=f'Примерная ROC (AUC ≈ {estimated_auc:.3f})')
                    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                           label='Случайный классификатор')
                    
                    ax.set_xlim([0.0, 1.0])
                    ax.set_ylim([0.0, 1.05])
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title('Примерная ROC Кривая')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'ROC данные недоступны', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def save_report(self):
        """Сохранение отчета"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Текстовые файлы", "*.txt"), ("JSON файлы", "*.json"), ("Все файлы", "*.*")],
                title="Сохранить отчет обучения"
            )
            
            if filename:
                if filename.endswith('.json'):
                    # JSON отчет
                    report_data = {
                        'user': self.user.username,
                        'training_date': datetime.now().isoformat(),
                        'training_results': self.results
                    }
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(report_data, f, indent=2, ensure_ascii=False)
                else:
                    # Текстовый отчет
                    report = self.generate_report()
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(report)
                
                messagebox.showinfo("Успех", f"Отчет сохранен: {filename}")
                
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка сохранения: {str(e)}")