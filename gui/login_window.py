import tkinter as tk
from tkinter import ttk, messagebox
import time
from typing import Callable, Optional

from models.user import User
from auth.password_auth import PasswordAuthenticator
from auth.keystroke_auth import KeystrokeAuthenticator
from config import FONT_FAMILY, FONT_SIZE, PANGRAM

class LoginWindow:
    """Окно входа с поэтапной двухфакторной аутентификацией"""
    
    def __init__(self, parent, password_auth: PasswordAuthenticator, 
                 keystroke_auth: KeystrokeAuthenticator, on_success: Callable):
        self.parent = parent
        self.password_auth = password_auth
        self.keystroke_auth = keystroke_auth
        self.on_success = on_success
        
        # Создание окна
        self.window = tk.Toplevel(parent)
        self.window.title("Вход в систему")
        self.window.geometry("600x750")
        self.window.resizable(True, True)
        self.window.minsize(550, 1000)
        
        # Модальное окно
        self.window.transient(parent)
        self.window.grab_set()
        
        # Центрирование
        self.center_window()
        
        # Переменные состояния
        self.current_user: Optional[User] = None
        self.session_id: Optional[str] = None
        self.is_recording = False
        self.login_phase = "credentials"  # "credentials" или "keystroke"
        
        # Нормализованный текст для сравнения
        self.normalized_target = self._normalize_text(PANGRAM)
        
        # Инициализируем переменные виджетов
        self.start_recording_btn = None
        self.complete_auth_btn = None
        self.pangram_entry = None
        self.keystroke_frame = None
        
        # Создание интерфейса
        self.create_widgets()
        self.username_entry.focus()
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста - убираем пробелы и приводим к нижнему регистру"""
        return text.lower().replace(" ", "")
    
    def center_window(self):
        """Центрирование окна"""
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
    
    def create_widgets(self):
        """Создание виджетов окна входа"""
        main_frame = ttk.Frame(self.window, padding=30)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Заголовок
        title_label = ttk.Label(
            main_frame,
            text="Вход в систему",
            font=(FONT_FAMILY, 18, 'bold')
        )
        title_label.pack(pady=(0, 20))
        
        # Этап 1: Ввод учетных данных
        self.credentials_frame = ttk.LabelFrame(
            main_frame,
            text="Этап 1: Ввод учетных данных",
            padding=20
        )
        self.credentials_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Имя пользователя
        ttk.Label(self.credentials_frame, text="Имя пользователя:").pack(anchor=tk.W, pady=(0, 5))
        self.username_entry = ttk.Entry(self.credentials_frame, width=30, font=(FONT_FAMILY, FONT_SIZE))
        self.username_entry.pack(fill=tk.X)
        
        # Пароль
        ttk.Label(self.credentials_frame, text="Пароль:").pack(anchor=tk.W, pady=(15, 5))
        self.password_entry = ttk.Entry(
            self.credentials_frame, 
            width=30, 
            show="*",
            font=(FONT_FAMILY, FONT_SIZE)
        )
        self.password_entry.pack(fill=tk.X)
        
        # Кнопка первого этапа
        self.check_credentials_btn = ttk.Button(
            self.credentials_frame,
            text="Проверить учетные данные",
            command=self.check_credentials
        )
        self.check_credentials_btn.pack(pady=(15, 0))
        
        # Статус первого этапа
        self.credentials_status = ttk.Label(
            self.credentials_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.credentials_status.pack(pady=(10, 0))
        
        # Этап 2: Проверка динамики нажатий (изначально скрыт)
        self.keystroke_frame = ttk.LabelFrame(
            main_frame,
            text="Этап 2: Проверка динамики нажатий",
            padding=20
        )
        
        # Инструкции
        instructions_text = f"""Теперь введите панграмму для проверки динамики нажатий:
        
"{PANGRAM}"

Правила ввода:
Печатайте в том же стиле, что и при обучении
Регистр букв не важен  
Пробелы можно пропускать
При ошибке ввод сбросится автоматически"""
        
        self.instructions_label = ttk.Label(
            self.keystroke_frame,
            text=instructions_text,
            wraplength=400,
            justify=tk.LEFT,
            font=(FONT_FAMILY, 10)
        )
        self.instructions_label.pack(pady=(0, 15))
        
        # Кнопка начала записи
        self.start_recording_btn = ttk.Button(
            self.keystroke_frame,
            text="Начать ввод панграммы",
            command=self.start_pangram_input
        )
        self.start_recording_btn.pack(pady=(0, 15))
        
        # Прогресс ввода
        self.typing_progress_label = ttk.Label(
            self.keystroke_frame,
            text="",
            font=(FONT_FAMILY, 9, 'italic'),
            foreground='gray'
        )
        self.typing_progress_label.pack()
        
        # Поле ввода панграммы
        self.pangram_entry = ttk.Entry(
            self.keystroke_frame,
            width=50,
            font=(FONT_FAMILY, FONT_SIZE),
            state=tk.DISABLED
        )
        self.pangram_entry.pack(pady=(10, 0))
        
        # Статус записи
        self.recording_status = ttk.Label(
            self.keystroke_frame,
            text="",
            font=(FONT_FAMILY, 10)
        )
        self.recording_status.pack(pady=(10, 0))
        
        # Кнопка завершения
        self.complete_auth_btn = ttk.Button(
            self.keystroke_frame,
            text="Завершить аутентификацию",
            command=self.complete_authentication,
            state=tk.DISABLED
        )
        self.complete_auth_btn.pack(pady=(15, 0))
        
        # Общие кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        cancel_btn = ttk.Button(
            button_frame,
            text="Отмена",
            command=self.window.destroy,
            width=15
        )
        cancel_btn.pack()
        
        # Привязка событий
        self.username_entry.bind('<Return>', lambda e: self.password_entry.focus())
        self.password_entry.bind('<Return>', lambda e: self.check_credentials())
        
        # Настройка записи нажатий
        self.setup_keystroke_recording()
    
    def check_credentials(self):
        """Проверка учетных данных (первый этап)"""
        username = self.username_entry.get().strip()
        password = self.password_entry.get()
        
        if not username or not password:
            messagebox.showerror("Ошибка", "Заполните имя пользователя и пароль")
            return
        
        # Обновление статуса
        self.credentials_status.config(text="Проверка учетных данных...", foreground="blue")
        self.check_credentials_btn.config(state=tk.DISABLED)
        self.window.update()
        
        # Проверка пароля
        success, message, user = self.password_auth.authenticate(username, password)
        
        if not success:
            self.credentials_status.config(text="Ошибка входа", foreground="red")
            self.check_credentials_btn.config(state=tk.NORMAL)
            messagebox.showerror("Ошибка", message)
            self.password_entry.delete(0, tk.END)
            return
        
        self.current_user = user
        
        # Если пользователь не обучен
        if not user.is_trained:
            self.credentials_status.config(text="Вход выполнен (только пароль)", foreground="green")
            messagebox.showinfo(
                "Информация",
                "Модель динамики нажатий не обучена.\nВход выполнен только по паролю."
            )
            self.on_success(user)
            self.window.destroy()
            return
        
        # Переход ко второму этапу
        self.credentials_status.config(text="Учетные данные проверены", foreground="green")
        self.show_keystroke_phase()
    
    def show_keystroke_phase(self):
        """Показать этап проверки динамики нажатий"""
        self.login_phase = "keystroke"
        
        # Блокируем поля первого этапа
        self.username_entry.config(state=tk.DISABLED)
        self.password_entry.config(state=tk.DISABLED)
        self.check_credentials_btn.config(state=tk.DISABLED)
        
        # Показываем второй этап
        self.keystroke_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Обновляем окно
        self.window.update()
        
        # Проверяем что кнопка создана
        if self.start_recording_btn:
            self.start_recording_btn.focus()
        
        messagebox.showinfo(
            "Переход ко второму этапу",
            "Учетные данные проверены.\n\n"
            "Введите фразу для проверки динамики нажатий."
        )
    
    def start_pangram_input(self):
        """Начало ввода панграммы"""
    
        # Активируем поле ввода
        self.pangram_entry.config(state=tk.NORMAL)
        self.pangram_entry.delete(0, tk.END)
        self.pangram_entry.focus()
    
        # Принудительно начинаем запись
        if self.current_user and not self.is_recording:
            try:
                self.session_id = self.keystroke_auth.start_keystroke_recording(self.current_user.id)
                self.is_recording = True
            except Exception as e:
                print(f"Ошибка принудительного начала записи: {e}")
    
        # Обновляем интерфейс
        self.start_recording_btn.config(state=tk.DISABLED)
        self.recording_status.config(
            text="Запись активна - печатайте панграмму",
            foreground="red"
        )
    
    def setup_keystroke_recording(self):
        """Настройка записи динамики нажатий"""
        # Привязываем обработчики после создания виджета
        def bind_events():
            if self.pangram_entry:
                self.pangram_entry.bind('<FocusIn>', self.on_pangram_focus_in)
                self.pangram_entry.bind('<FocusOut>', self.on_pangram_focus_out)
                self.pangram_entry.bind('<KeyPress>', self.on_key_press)
                self.pangram_entry.bind('<KeyRelease>', self.on_key_release)
                self.pangram_entry.bind('<KeyRelease>', self.check_pangram_input, add='+')
        
        # Отложенная привязка после создания всех виджетов
        self.window.after(100, bind_events)
    
    def on_pangram_focus_in(self, event=None):
        """Начало записи в поле панграммы"""
        if self.login_phase == "keystroke" and not self.is_recording and self.current_user:
            try:
                self.session_id = self.keystroke_auth.start_keystroke_recording(self.current_user.id)
                self.is_recording = True
                self.recording_status.config(
                    text="Запись динамики активна - печатайте панграмму",
                    foreground="red"
                )
            except Exception as e:
                self.recording_status.config(
                    text="Ошибка начала записи",
                    foreground="red"
                )
    
    def on_pangram_focus_out(self, event=None):
        """Не останавливаем запись"""
        if self.is_recording:
            # Не останавливаем запись, только меняем индикатор
            self.recording_status.config(
                text="Запись продолжается (можно продолжать печатать)",
                foreground="orange"
            )
    
    def on_key_press(self, event):
        """Обработка нажатия клавиши"""
        if self.is_recording and self.session_id:
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(
                    self.session_id,
                    event.keysym,
                    'press'
                )
    
    def on_key_release(self, event):
        """Обработка отпускания клавиши"""
        if self.is_recording and self.session_id:
            if event.keysym not in ['Shift_L', 'Shift_R', 'Control_L', 'Control_R', 
                                   'Alt_L', 'Alt_R', 'Caps_Lock', 'Tab']:
                self.keystroke_auth.record_key_event(
                    self.session_id,
                    event.keysym,
                    'release'
                )
    
    def check_pangram_input(self, event=None):
        """Проверка ввода панграммы в реальном времени"""
        if self.login_phase != "keystroke":
            return
        
        current_text = self.pangram_entry.get()
        normalized_current = self._normalize_text(current_text)
    
        # Проверяем длину
        if len(normalized_current) > len(self.normalized_target):
            self._reset_pangram_input("Текст слишком длинный. Начните заново.")
            return
    
        # Проверяем правильность префикса
        is_correct_prefix = True
        for i, char in enumerate(normalized_current):
            if i >= len(self.normalized_target) or char != self.normalized_target[i]:
                is_correct_prefix = False
                break
    
        if not is_correct_prefix:
            self._reset_pangram_input("Ошибка в тексте. Начните заново.")
            return
    
        # Если записи нет, но текст вводится - начинаем запись принудительно
        if not self.is_recording and len(normalized_current) > 0:
            self.on_pangram_focus_in()
    
        # Обновляем прогресс
        if len(normalized_current) > 0:
            progress_text = f"Введено: {len(normalized_current)}/{len(self.normalized_target)} символов"
            if len(current_text) > 0:
                progress_text += f" | '{current_text[-min(8, len(current_text)):]}'"
            self.typing_progress_label.config(text=progress_text)
        else:
            self.typing_progress_label.config(text="")
    
        # Проверяем завершенность
        if normalized_current == self.normalized_target:
            self.complete_auth_btn.config(state=tk.NORMAL)
            self.recording_status.config(
                text="Панграмма введена полностью! Запись завершена. Можно завершить аутентификацию.",
                foreground="green"
            )
        else:
            self.complete_auth_btn.config(state=tk.DISABLED)
            if len(normalized_current) > 0:
                self.recording_status.config(
                    text="Продолжайте ввод панграммы...",
                    foreground="red"
                )
    
    def _reset_pangram_input(self, message: str):
        """Сброс ввода панграммы при ошибке"""
        # Останавливаем запись
        if self.is_recording:
            self.is_recording = False
            if self.session_id and self.session_id in self.keystroke_auth.current_session:
                del self.keystroke_auth.current_session[self.session_id]
            self.session_id = None
        
        # Очищаем поле
        self.pangram_entry.delete(0, tk.END)
        
        # Показываем ошибку
        self.recording_status.config(text=message, foreground="red")
        self.typing_progress_label.config(text="")
        if self.complete_auth_btn:
            self.complete_auth_btn.config(state=tk.DISABLED)
        
        # Через 2 секунды перезапускаем
        self.window.after(2000, self._restart_pangram_input)
    
    def _restart_pangram_input(self):
        """Перезапуск ввода панграммы"""
        self.recording_status.config(
            text="Нажмите в поле ввода и начните печатать панграмму заново",
            foreground="blue"
        )
        self.pangram_entry.focus()
    
    def complete_authentication(self):
        """Завершение аутентификации"""
    
        # Проверяем состояние
        if not self.session_id:
            messagebox.showerror("Ошибка", "Отсутствует ID сессии записи")
            return
        
        if self.login_phase != "keystroke":
            messagebox.showerror("Ошибка", "Неправильная фаза аутентификации")
            return
        
        if not self.current_user:
            messagebox.showerror("Ошибка", "Пользователь не определен")
            return
    
        # Финальная проверка панграммы
        current_text = self.pangram_entry.get()
        normalized_current = self._normalize_text(current_text)
    
        if normalized_current != self.normalized_target:
            messagebox.showerror("Ошибка", "Панграмма введена не полностью или неправильно")
            return
    
        # Принудительно проверяем, что запись была
        if self.session_id not in self.keystroke_auth.current_session:
            messagebox.showerror(
                "Ошибка",
                "Сессия записи потеряна. Попробуйте ввести панграмму заново."
            )
            self._restart_for_retry()
            return
    
        # Завершаем запись
        self.complete_auth_btn.config(state=tk.DISABLED, text="Анализ...")
        self.recording_status.config(text="Анализ динамики нажатий...", foreground="blue")
        self.window.update()
    
        try:
            # Принудительно устанавливаем, что запись активна
            self.is_recording = True
        
            # Получаем признаки
            features = self.keystroke_auth.finish_recording(self.session_id)
        
            # Проверяем качество записи
            if not features or all(v == 0 for v in features.values()):
                messagebox.showerror(
                    "Ошибка",
                    "Не удалось записать динамику нажатий.\n"
                    "Попробуйте ввести панграмму еще раз."
                )
                self._restart_for_retry()
                return
        
            # Аутентификация
            auth_success, confidence, auth_message = self.keystroke_auth.authenticate(
                self.current_user, 
                features
            )
        
        
            if auth_success:
                self.recording_status.config(
                    text=f"{auth_message}",
                    foreground="green"
                )
                self.window.update()
                time.sleep(1)
            
                messagebox.showinfo(
                    "Успех",
                    f"Аутентификация успешна!\nУверенность: {confidence:.1%}"
                )
            
                self.on_success(self.current_user)
                self.window.destroy()
            else:
                self.recording_status.config(text="Аутентификация отклонена", foreground="red")
                messagebox.showerror(
                    "Ошибка аутентификации",
                    f"{auth_message}\n\n"
                    f"Уверенность: {confidence:.1%}\n"
                    "Попробуйте ввести панграмму в том же стиле, что и при обучении."
                )
                self._restart_for_retry()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Ошибка", f"Ошибка при анализе: {str(e)}")
            self._restart_for_retry()
    
    def _restart_for_retry(self):
        """Перезапуск для повторной попытки"""
        # Сбрасываем состояние
        self.session_id = None
        self.is_recording = False
        
        # Очищаем поле
        self.pangram_entry.delete(0, tk.END)
        
        # Возвращаем кнопки в исходное состояние
        self.complete_auth_btn.config(state=tk.DISABLED, text="Завершить аутентификацию")
        self.start_recording_btn.config(state=tk.NORMAL)
        self.pangram_entry.config(state=tk.DISABLED)
        
        # Обновляем статус
        self.recording_status.config(
            text="Попробуйте еще раз - нажмите 'Начать ввод панграммы'",
            foreground="black"
        )
        self.typing_progress_label.config(text="")
        
        # Фокус на кнопку
        self.start_recording_btn.focus()