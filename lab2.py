import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Параметры времени и шагов
num_steps = 1000
time = np.linspace(0, 10, num_steps)

# Углы
angle1 = np.cos(time)
angle2 = np.sin(time)

# Радиусы и параметры системы
radius_main = 1
radius_inner1 = 0.125
radius_inner2 = 0.05
a = radius_main - radius_inner2
b = radius_main - radius_inner1

# Анимация
fig = plt.figure()
axis = fig.add_subplot(1, 1, 1)
axis.axis('equal')

def rotate_2D(x, y, angle):
    new_x = x * np.cos(angle) - y * np.sin(angle)
    new_y = x * np.sin(angle) + y * np.cos(angle)
    return new_x, new_y

# Центр окружности
center_x, center_y = 0, 1

# Центральный элемент
central_circle = plt.Circle((center_x, center_y), radius_inner2, color='black')

# Координаты первого подвижного элемента
circle1_x = np.zeros(num_steps)
circle1_y = np.zeros(num_steps)

for i in range(len(angle1)):
    new_x, new_y = rotate_2D(0, -a, angle1[i])
    circle1_x[i] = center_x + new_x
    circle1_y[i] = center_y + new_y

# Координаты второго подвижного элемента
circle2_x = np.zeros(num_steps)
circle2_y = np.zeros(num_steps)

for i in range(len(angle1)):
    new_x, new_y = rotate_2D(0, -b, angle2[i])
    circle2_x[i] = circle1_x[i] + new_x
    circle2_y[i] = circle1_y[i] + new_y

# Добавление центрального круга
axis.add_patch(central_circle)

# Добавление двух движущихся кругов
moving_circle1 = plt.Circle((circle1_x[0], circle1_y[0]), radius_main, color='black', fill=False)
moving_circle2 = plt.Circle((circle2_x[0], circle2_y[0]), radius_inner1, color='black', fill=False)

axis.add_patch(moving_circle1)
axis.add_patch(moving_circle2)

# Функция анимации
def animate(i):
    moving_circle1.center = (circle1_x[i], circle1_y[i])
    moving_circle2.center = (circle2_x[i], circle2_y[i])

axis.set(xlim=[-2, 2], ylim=[-2, 2])


animation = FuncAnimation(fig, animate, frames=num_steps, interval=1)

plt.show()
