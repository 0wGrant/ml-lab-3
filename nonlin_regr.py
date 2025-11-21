import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

# Генерация данных
def true_function(x):
    return x**3

num_train = 500
noise_level = 0.1

X_train = np.linspace(-0.2, 1.2, num_train).reshape(-1, 1)
y_train = true_function(X_train) + np.random.normal(0, noise_level, (num_train, 1))

X_test = np.linspace(-0.5, 1.5, 300).reshape(-1, 1)
y_test = true_function(X_test)

# Архитектура с Leaky ReLU
model = keras.Sequential([
    layers.Dense(15, kernel_initializer='he_normal'),
    layers.LeakyReLU(negative_slope=0.01),

    layers.Dense(20, activation='tanh', kernel_initializer='glorot_normal'),

    layers.Dense(1)
])

model.compile(
    loss='mse',
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=['mae']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=100,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=2000,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

# Предсказание
y_pred = model.predict(X_test).flatten()

# Визуализация
plt.figure(figsize=(16, 6))

# Основной график
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, s=10, alpha=0.4, color='grey', label='Данные (+шум)')
plt.plot(X_test, y_test, 'g--', lw=2, label='Истинная $y = x^3$')
plt.plot(X_test, y_pred, 'r', lw=2.5, label='Предсказание')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Модель остановлена на эпохе {early_stop.stopped_epoch}')
plt.legend()
plt.grid(True)
plt.xlim(-0.5, 1.5)
plt.ylim(-2, 3)

# График ошибок
plt.subplot(1, 2, 2)
plt.semilogy(history.history['loss'], label='Ошибка обучения')
plt.semilogy(history.history['val_loss'], label='Ошибка валидации')
plt.axvline(early_stop.stopped_epoch, color='k', linestyle='--', label='Ранняя остановка')
plt.xlabel('Эпоха')
plt.ylabel('MSE (лог. шкала)')
plt.title(f'Лучшая Val MSE: {np.min(history.history["val_loss"]):.4f}')
plt.legend()
plt.grid(True, which='both')

plt.tight_layout()
plt.show()