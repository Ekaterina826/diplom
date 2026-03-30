"""
Модуль для обучения модели машинного обучения на фитнес-данных.
Используется Random Forest с балансировкой классов.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib


def prepare_preprocessor(numerical_features: list, categorical_features: list):
    """
    Создаёт препроцессор для числовых и категориальных признаков.

    Args:
        numerical_features (list): Список имён числовых признаков.
        categorical_features (list): Список имён категориальных признаков.

    Returns:
        ColumnTransformer: Настроенный препроцессор.
    """
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )


def train_fitness_model(csv_path: str, model_save_path: str = 'models/model.pkl',
                        preproc_save_path: str = 'models/preprocessor.pkl'):
    """
    Обучает модель классификации уровня подготовки на основе фитнес-данных.

    Args:
        csv_path (str): Путь к CSV-файлу с данными.
        model_save_path (str): Путь для сохранения обученной модели.
        preproc_save_path (str): Путь для сохранения препроцессора.

    Returns:
        None
    """
    # Загрузка данных
    df = pd.read_csv(csv_path)
    y = df['Experience_Level']
    features = [
        'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
        'Workout_Type', 'Calories_Burned', 'Session_Duration (hours)',
        'Avg_BPM', 'Workout_Frequency (days/week)'
    ]
    X = df[features]

    # Разделение признаков
    numerical = ['Age', 'Weight (kg)', 'Height (m)', 'BMI',
                 'Calories_Burned', 'Session_Duration (hours)',
                 'Avg_BPM', 'Workout_Frequency (days/week)']
    categorical = ['Gender', 'Workout_Type']

    # Подготовка препроцессора и преобразование данных
    preprocessor = prepare_preprocessor(numerical, categorical)
    X_processed = preprocessor.fit_transform(X)

    # Обучение модели
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight='balanced'  # балансировка несбалансированных классов
    )
    model.fit(X_processed, y)

    # Сохранение
    joblib.dump(model, model_save_path)
    joblib.dump(preprocessor, preproc_save_path)
    print(f" Модель сохранена: {model_save_path}")
    print(f" Препроцессор сохранён: {preproc_save_path}")
