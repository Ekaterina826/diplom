# СРАВНЕНИЕ МЕТОДОВ МАШИННОГО ОБУЧЕНИЯ ДЛЯ АНАЛИЗА ФИТНЕС-ДАННЫХ

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
import time

# загружаем данные
df = pd.read_csv('gym_members_exercise_tracking.csv')

# ПОДГОТОВКА ДАННЫХ
# Целевая переменная
y = df['Experience_Level']

# Признаки для анализа
features = [
    'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
    'Workout_Type', 'Calories_Burned', 'Session_Duration (hours)',
    'Avg_BPM', 'Workout_Frequency (days/week)'
]
X = df[features]

# Разделение на числовые и категориальные признаки
numerical_features = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 
                     'Calories_Burned', 'Session_Duration (hours)', 
                     'Avg_BPM', 'Workout_Frequency (days/week)']
categorical_features = ['Gender', 'Workout_Type']

# Препроцессор
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ОПРЕДЕЛЕНИЕ МОДЕЛЕЙ
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced' 
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42, 
        class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced'
    ),
    'SVM': SVC(
        kernel='rbf', 
        gamma='scale', 
        random_state=42, 
        class_weight='balanced' 
    ),
    'MLP': MLPClassifier(
        hidden_layer_sizes=(50,), 
        max_iter=500, 
        random_state=42
    )
}

# ОБУЧЕНИЕ И ОЦЕНКА 
print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ МАШИННОГО ОБУЧЕНИЯ")
print(f"{'Метод':<20} | {'Accuracy':<10} | {'F1-score (macro)':<15} | {'Время (с)':<10}")

best_model = None
best_score = 0

for name, model in models.items():
    # Создание pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
    
    # Обучение
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Предсказание
    y_pred = pipeline.predict(X_test)
    
    # Метрики
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    # Вывод строки
    print(f"{name:<20} | {acc:<10.3f} | {f1:<15.3f} | {train_time:<10.3f}")
    
    # Определяем лучшую модель по Accuracy
    if acc > best_score:
        best_score = acc
        best_model = name

print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model} (Accuracy = {best_score:.3f})")
print("\nВывод: Для анализа фитнес-данных наиболее эффективен метод", best_model) 
