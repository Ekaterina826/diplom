"""
Модуль содержит обработчики сообщений для Telegram-бота.
Реализует пошаговый сбор фитнес-данных от пользователя.
"""

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import ContextTypes
import joblib
import numpy as np

# Глобальное хранилище данных пользователя (в реальном проекте лучше использовать context.user_data)
user_data = {}


async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обработчик команды /start. Начинает диалог.

    Args:
        update: Объект обновления от Telegram API.
        context: Контекст выполнения (используется для хранения состояния).

    Returns:
        int: Состояние AGE (следующий шаг диалога).
    """
    await update.message.reply_text(
        "Привет! Я помогу определить ваш уровень физической подготовки.\n"
        "Введите ваш возраст (полных лет):"
    )
    return 0  # AGE = 0


async def age_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Обрабатывает ввод возраста пользователем.

    Args:
        update: Объект обновления.
        context: Контекст.

    Returns:
        int: Следующее состояние (GENDER = 1) или повтор AGE при ошибке.
    """
    try:
        age_val = int(update.message.text)
        if not (10 <= age_val <= 100):
            raise ValueError
        user_data['Age'] = age_val
        reply_keyboard = [['Мужской', 'Женский']]
        await update.message.reply_text(
            "Укажите пол:",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return 1  # GENDER
    except ValueError:
        await update.message.reply_text("Пожалуйста, введите корректный возраст (10–100).")
        return 0  # AGE
