import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df = pd.read_csv('data/figures.csv')

# 3 обираємо елементи для навчання
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])

# Додаємо нову ознаку
df['ratio'] = df['area'] / df['perimeter']

# Вхідні дані
X = df[['area', 'perimeter', 'corners', 'ratio']]
y = df['label_enc']

num_classes = len(df['label_enc'].unique())

# 4 створюємо модель
model = keras.Sequential([
    layers.Dense(16, activation="relu", input_shape=(4,)),   # ← 16 нейронів
    layers.Dense(16, activation="relu"),                      # ← 8 нейронів
    layers.Dense(num_classes, activation="softmax")          # ← кількість класів
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# epochs можна змінювати
history = model.fit(X, y, epochs=500, verbose=0)

# візуалізація навчання
plt.plot(history.history['loss'], label='Втрата(loss)')
plt.plot(history.history['accuracy'], label='Точність(accuracy)')
plt.xlabel("Epoch")
plt.ylabel("Znachenna")
plt.title('study process')
plt.legend()
plt.show()

# тестування
test = np.array([[18, 16, 0, 18/16]])  # Додано ratio
pred = model.predict(test)

print(f'Імовірність по кожному класу: {pred}')
print(f'модель визначила: {encoder.inverse_transform([np.argmax(pred)])}')
