import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping


# Генерация данных
def true_decision_boundary(x):
    return 2.5 * x - 1.2  # Истинная разделяющая граница


num_samples = 200
noise_level = 0.5

# Генерация данных с проверкой классов
while True:
    X = np.random.uniform(-6, 3, (num_samples, 2))
    y = (X[:, 1] > true_decision_boundary(X[:, 0]) +
         np.random.normal(0, noise_level, num_samples)).astype(int)
    if len(np.unique(y)) > 1:
        break

# Разделение данных (сохраняем пропорцию 80/20)
split_idx = int(0.8 * num_samples)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Модель
model = keras.Sequential([
    layers.Dense(1, activation='sigmoid', input_shape=(2,))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.02),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.25,
    callbacks=[early_stop],
    verbose=0
)

# Визуализация
plt.figure(figsize=(14, 6))

# График данных
plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
            cmap='bwr', alpha=0.8, edgecolors='k',
            label='Обучающая выборка')

# Построение разделяющей границы
x_grid = np.linspace(-2, 2, 100)
y_grid = np.linspace(-6, 6, 100)
xx, yy = np.meshgrid(x_grid, y_grid)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()], verbose=0)
Z = Z.reshape(xx.shape)

if not np.all(Z == Z[0, 0]):
    plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=3)

# Истинная граница и настройки графика
plt.plot(x_grid, true_decision_boundary(x_grid), 'k--', label='Истинная граница')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Линейная классификация (только обучающие данные)')
plt.legend()
plt.grid(True)

# График обучения
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Точность (обучение)')
plt.plot(history.history['val_accuracy'], label='Точность (валидация)')
if early_stop.stopped_epoch > 0:
    plt.axvline(early_stop.stopped_epoch, color='k', linestyle='--', label='Ранняя остановка')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.title(f'Лучшая точность: {np.max(history.history["val_accuracy"]):.2%}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Вывод информации
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nТестовая точность: {test_acc:.2%}")
print("Коэффициенты модели:")
print(f"Weights: {model.layers[0].weights[0].numpy().flatten()}")
print(f"Bias: {model.layers[0].bias.numpy()[0]}")