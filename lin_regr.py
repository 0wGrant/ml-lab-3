import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

# Генерация данных
def true_function(x):
    return 2.5 * x + 1.8

num_samples = 100
noise_level = 0.8

# Генерация данных
X = np.random.uniform(-3, 3, num_samples)
y = true_function(X) + np.random.normal(0, noise_level, num_samples)

# Разделение данных
split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Построение модели
model = keras.Sequential([
    layers.Dense(1, input_shape=(1,))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.1),
    loss='mse',
    metrics=['mae']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.25,
    callbacks=[early_stop],
    verbose=0
)

# Предсказание
x_range = np.linspace(-3, 3, 100)
y_pred = model.predict(x_range.reshape(-1, 1)).flatten()

# Визуализация
plt.figure(figsize=(14, 6))

# График данных и линии регрессии
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Обучающие данные')
plt.scatter(X_test, y_test, color='grey', alpha=0.4, label='Тестовые данные')
plt.plot(x_range, true_function(x_range), 'k--', label='Истинная зависимость')
plt.plot(x_range, y_pred, 'r-', linewidth=2, label='Предсказание модели')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Линейная регрессия')
plt.legend()
plt.grid(True)

# График обучения
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Ошибка (обучение)')
plt.plot(history.history['val_loss'], label='Ошибка (валидация)')
if early_stop.stopped_epoch > 0:
    plt.axvline(early_stop.stopped_epoch, color='k', linestyle='--', label='Ранняя остановка')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.title(f'Лучшая MSE: {np.min(history.history["val_loss"]):.2f}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод информации
weights = model.layers[0].weights[0].numpy()[0][0]
bias = model.layers[0].bias.numpy()[0]
print(f"\nПараметры модели:")
print(f"Коэффициент (наклон): {weights:.2f} (истинный 2.5)")
print(f"Смещение: {bias:.2f} (истинное 1.8)")
print(f"Тестовая MSE: {model.evaluate(X_test, y_test, verbose=0)[0]:.2f}")