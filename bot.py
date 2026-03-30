import logging
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)
import joblib
import numpy as np

# Включаем логирование
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Этапы диалога
(
    AGE, GENDER, WEIGHT, HEIGHT, WORKOUT_TYPE,
    SESSION_DURATION, FREQUENCY, CALORIES
) = range(8)

# Загрузка модели и препроцессора
try:
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    raise RuntimeError("Файлы model.pkl и preprocessor.pkl не найдены. Сначала запустите train_model.py!")

# Глобальный словарь для хранения данных пользователя в рамках сессии
user_data = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! \nЯ помогу определить ваш уровень физической подготовки на основе ваших данных.\n\n"
        "Введите ваш возраст (полных лет, например: 25):"
    )
    return AGE

async def age(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        age_val = int(update.message.text)
        if not (10 <= age_val <= 100):
            raise ValueError
        user_data['Age'] = age_val
        reply_keyboard = [['Мужской', 'Женский']]
        await update.message.reply_text(
            "Укажите ваш пол:",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return GENDER
    except (ValueError, TypeError):
        await update.message.reply_text("Пожалуйста, введите корректный возраст (число от 10 до 100).")
        return AGE

async def gender(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text in ['Мужской', 'Женский']:
        user_data['Gender'] = text
        await update.message.reply_text("Введите ваш вес в килограммах (например: 65.5):")
        return WEIGHT
    else:
        await update.message.reply_text("Пожалуйста, выберите пол из кнопок.")
        return GENDER

async def weight(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        weight_val = float(update.message.text)
        if weight_val <= 0 or weight_val > 300:
            raise ValueError
        user_data['Weight (kg)'] = weight_val
        await update.message.reply_text("Введите ваш рост в метрах (например: 1.75):")
        return HEIGHT
    except (ValueError, TypeError):
        await update.message.reply_text("Пожалуйста, введите корректный вес (положительное число, например: 60.5).")
        return WEIGHT

async def height(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        height_val = float(update.message.text)
        if height_val <= 0 or height_val > 3:
            raise ValueError
        user_data['Height (m)'] = height_val
        # Автоматический расчёт ИМТ
        bmi = round(weight_val / (height_val ** 2), 2)
        user_data['BMI'] = bmi
        reply_keyboard = [['Cardio', 'Strength', 'Yoga', 'HIIT']]
        await update.message.reply_text(
            f"Ваш ИМТ: {bmi}\n\nВыберите тип тренировки:",
            reply_markup=ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
        )
        return WORKOUT_TYPE
    except (ValueError, TypeError, NameError):
        await update.message.reply_text("Пожалуйста, введите корректный рост (например: 1.70).")
        return HEIGHT

async def workout_type(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if text in ['Cardio', 'Strength', 'Yoga', 'HIIT']:
        user_data['Workout_Type'] = text
        await update.message.reply_text("Длительность одной тренировки в часах (например: 1.5):")
        return SESSION_DURATION
    else:
        await update.message.reply_text("Пожалуйста, выберите тип тренировки из кнопок.")
        return WORKOUT_TYPE

async def session_duration(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sd = float(update.message.text)
        if sd <= 0 or sd > 5:
            raise ValueError
        user_data['Session_Duration (hours)'] = sd
        await update.message.reply_text("Сколько раз в неделю вы тренируетесь? (целое число, например: 3):")
        return FREQUENCY
    except (ValueError, TypeError):
        await update.message.reply_text("Введите длительность от 0.1 до 5 часов.")
        return SESSION_DURATION

async def frequency(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        freq = int(update.message.text)
        if not (1 <= freq <= 7):
            raise ValueError
        user_data['Workout_Frequency (days/week)'] = freq
        await update.message.reply_text("Примерное количество сожжённых калорий за одну тренировку:")
        return CALORIES
    except (ValueError, TypeError):
        await update.message.reply_text("Введите целое число от 1 до 7.")
        return FREQUENCY

async def calories(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        cal = float(update.message.text)
        if cal < 50 or cal > 2000:
            raise ValueError
        user_data['Calories_Burned'] = cal
        # Avg_BPM — можно запросить, но для упрощения используем среднее значение
        user_data['Avg_BPM'] = 120

        # Порядок признаков должен совпадать с обучением!
        feature_order = [
            'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
            'Workout_Type', 'Calories_Burned', 'Session_Duration (hours)',
            'Avg_BPM', 'Workout_Frequency (days/week)'
        ]
        input_data = [[user_data[feat] for feat in feature_order]]

        # Преобразование и прогноз
        X_transformed = preprocessor.transform(input_data)
        prediction = model.predict(X_transformed)[0]

        # Интерпретация результата
        levels = {1: "новичок", 2: "средний", 3: "эксперт"}
        recommendations = {
            1: "Рекомендуем начать с лёгких тренировок 2–3 раза в неделю. Не забывайте про разминку!",
            2: "Вы на хорошем уровне! Попробуйте добавить интервальные нагрузки или силовые упражнения.",
            3: "Отличный результат! Вы можете экспериментировать с продвинутыми программами и целями."
        }

        await update.message.reply_text(
            f" Ваш уровень подготовки: **{levels[prediction]}**\n\n"
            f"{recommendations[prediction]}",
            parse_mode='Markdown'
        )
        return ConversationHandler.END
    except (ValueError, TypeError):
        await update.message.reply_text("Введите реалистичное число калорий (от 50 до 2000).")
        return CALORIES

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Анализ отменён. Чтобы начать заново, отправьте /start.")
    return ConversationHandler.END

def main():
    #  Вставьте сюда ваш токен от @BotFather
    TOKEN = "ВАШ_ТОКЕН"

    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            AGE: [MessageHandler(filters.TEXT & ~filters.COMMAND, age)],
            GENDER: [MessageHandler(filters.TEXT & ~filters.COMMAND, gender)],
            WEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, weight)],
            HEIGHT: [MessageHandler(filters.TEXT & ~filters.COMMAND, height)],
            WORKOUT_TYPE: [MessageHandler(filters.TEXT & ~filters.COMMAND, workout_type)],
            SESSION_DURATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, session_duration)],
            FREQUENCY: [MessageHandler(filters.TEXT & ~filters.COMMAND, frequency)],
            CALORIES: [MessageHandler(filters.TEXT & ~filters.COMMAND, calories)],
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == '__main__':
    main()
