import numpy as np
from math import pi
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import math

t_fin = 20
t = np.linspace(0, t_fin, 1001)

s = 0.8 * np.sin(5 * t)
phi = 5 * np.sin(2*t)

x0 = 8
R = 1

angles = np.linspace(0, 2 * pi, 360)

box_w = 0.4
box_h = 0.2

spting_steps = 100

'''
    Для создания пружины
'''
def spring(k, h, w):
    x = np.linspace(0, h, spting_steps)
    return np.array([x, np.sin(2 * math.pi / (h / k) * x) * w])

'''
    Отрисовываем грузик
'''
box_x_tmp = np.array([-box_h / 2, -box_h / 2, box_h / 2, box_h / 2, -box_h / 2])
box_y_tmp = np.array([-box_w / 2, box_w / 2, box_w / 2, -box_w / 2, -box_w / 2])

'''
    Заполняем всё нулями
'''
ring_dots_x = np.zeros([len(t), len(angles)])
ring_dots_y = np.zeros([len(t), len(angles)])

box_dots_x = np.zeros([len(t), 5])
box_dots_y = np.zeros([len(t), 5])

spring_a_x = np.zeros([len(t), spting_steps])
spring_a_y = np.zeros([len(t), spting_steps])
spring_b_x = np.zeros([len(t), spting_steps])
spring_b_y = np.zeros([len(t), spting_steps])

spring_c_x = np.zeros([len(t), spting_steps])
spring_c_y = np.zeros([len(t), spting_steps])


for i in range(len(t)):
    '''
        Сам обруч
    '''
    ring_x = x0 + phi[i] * R
    ring_y = R

    ring_dots_x[i] = np.cos(phi[i]) * R * np.cos(angles) + np.sin(phi[i]) * R * np.sin(angles) + ring_x
    ring_dots_y[i] = - np.sin(phi[i]) * R * np.cos(angles) + np.cos(phi[i]) * R * np.sin(angles) + ring_y

    '''
        Грузик
    '''
    bx = box_x_tmp - s[i]
    by = box_y_tmp
    box_dots_x[i] = np.cos(phi[i]) * bx + np.sin(phi[i]) * by + ring_x
    box_dots_y[i] = - np.sin(phi[i]) * bx + np.cos(phi[i]) * by + ring_y

    '''
        Пружинка от стены к обручу
    '''
    spring_a_x[i] = spring(5, ring_x, 0.2)[0]
    spring_a_y[i] = spring(5, ring_x, 0.2)[1] + ring_y

    '''
        Пружинки внутри обруча
    '''
    b_x = R - spring(10, R + s[i] - box_h / 2, 0.16)[0]
    b_y = spring(10, R - s[i], 0.16)[1]
    spring_b_x[i] = np.cos(phi[i]) * b_x + np.sin(phi[i]) * b_y + ring_x
    spring_b_y[i] = -np.sin(phi[i]) * b_x + np.cos(phi[i]) * b_y + ring_y

    c_x = spring(10, R - s[i] - box_h / 2, 0.16)[0] - R
    c_y = spring(10, R - s[i], 0.16)[1]
    spring_c_x[i] = np.cos(phi[i]) * c_x + np.sin(phi[i]) * c_y + ring_x
    spring_c_y[i] = -np.sin(phi[i]) * c_x + np.cos(phi[i]) * c_y + ring_y

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")

surface = ax.plot([0, 0, 15], [5, 0, 0], "black")
ring, = ax.plot(ring_dots_x[0], ring_dots_y[0], "black")
box, = ax.plot(box_dots_x[0], box_dots_y[0], "black")
spring_a, = ax.plot(spring_a_x[0], spring_a_y[0], "red")
spring_b, = ax.plot(spring_b_x[0], spring_b_y[0], "purple")
spring_c, = ax.plot(spring_c_x[0], spring_c_y[0], "brown")

def animate(i):
    ring.set_data(ring_dots_x[i], ring_dots_y[i])
    box.set_data(box_dots_x[i], box_dots_y[i])
    spring_a.set_data(spring_a_x[i], spring_a_y[i])
    spring_b.set_data(spring_b_x[i], spring_b_y[i])
    spring_c.set_data(spring_c_x[i], spring_c_y[i])

    return ring, box, spring_a, spring_b, spring_c

animation = FuncAnimation(fig, animate, frames=1000, interval=60)
plt.show()