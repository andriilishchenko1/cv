import cv2
import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from numpy.f2py.crackfortran import verbose
from scipy.ndimage import label
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics
from tensorflow.python.layers.core import Dense

from Praktichna_2 import perimeter, shape

df = pd.read_csv('data/figures.csv')
#3 обираємо елементи для навчання
encoder = LabelEncoder()
df['label_enc'] = encoder.fit_transform(df['label'])
X = df[['area', 'perimeter', 'corners']]
y = df['label_enc']
#4 створюємо модель

model = keras.Sequential([layers.Dense(8, activation = "relu", input_shape = (3, )),
                          layers.Dense(8, activation = "relu"),
                          layers.Dense(8, activation="softmax")])# кількість нейронів у шарі(8)


model.compile(optimizer = 'adam',lose = 'binary_crossentropy', metrics = ['accuracy'] )
history = model.fit(X, y, epochs = 200, verbose = 0)


# візуалізація навчання
plt.plot(history.history['loss'], label = 'Втрата(loss)')
plt.plot(history.history['accuracy'], label = 'Точність(accuracy)')
plt.xlabel("Epoch")
plt.ylabel("Znachenna")
plt.title('study process')
plt.legend()
plt.show()




#тестування

test = np.array([18, 16, 0])
pred = model.predict(test)
print(f'Імовірність по кожному класу:{pred}')
print(f'модель визначила:{encoder.inverse_transform([np.argmax(pred)])}')









