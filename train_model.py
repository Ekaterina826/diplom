import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Загрузка данных
df = pd.read_csv('gym_members_exercise_tracking.csv')

# 2. Целевая переменная и признаки
y = df['Experience_Level']
features = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
    'Workout_Type', 'Calories_Burned', 'Session_Duration (hours)',
    'Avg_BPM', 'Workout_Frequency (days/week)'
]
X = df[features]

# 3. Разделение на числовые и категориальные признаки
numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'BMI',
                     'Calories_Burned', 'Session_Duration (hours)',
                     'Avg_BPM', 'Workout_Frequency (days/week)']
categorical_features = ['Gender', 'Workout_Type']

# 4. Создание препроцессора
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# 5. Подготовка данных
X_processed = preprocessor.fit_transform(X)

# 6. Обучение модели
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight='balanced'  # балансировка классов
)
model.fit(X_processed, y)

# 7. Сохранение модели и препроцессора
joblib.dump(model, 'model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

print("Модель и препроцессор сохранены!")
print("Файлы: model.pkl, preprocessor.pkl")
