#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np


def visualize_classifier(classifier, X, y, title=''):
    # Границы области построения
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Шаг сетки
    mesh_step_size = 0.01

    # Сетка точек
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )

    # Предсказание класса для каждой точки сетки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
    output = output.reshape(x_vals.shape)

    # Рисуем области решений
    plt.figure()
    plt.title(title)
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')

    # Рисуем исходные точки
    plt.scatter(
        X[:, 0], X[:, 1], c=y, s=75,
        edgecolors='black', linewidth=1, cmap=plt.cm.Paired
    )

    # Настройки осей
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())
    plt.xticks(np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0))
    plt.yticks(np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0))

    plt.show()