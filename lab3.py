import numpy as np
from math import pi
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt
from scipy.integrate import odeint
import math

'''
    y - Вектор состояния системы
    t - Время
    m1 - Масса обруча
    m2 - Масса грузика
    c - Жесткость пружин для грузика
    c1 - Жесткость пружины для обруча
'''
def fnc(y, t, m1, m2, c, c1, R, g):
    dy = np.zeros(4)
    dy[0] = y[2]
    dy[1] = y[3]

    a11 = -m2 * R * np.cos(y[1])
    a12 = ((2 * m1 + m2) * R ** 2 + m2 * y[0]**2 + 2 * R * m2 * y[0] * np.sin(y[1]))
    a21 = 1
    a22 = -R * np.cos(y[1])

    b1 = m2 * R * y[0] * y[3] ** 2 * np.cos(y[1]) - 2 * m2 * (y[0] + R * np.sin(y[1])) * y[2] * y[3] - c1 * R ** 2 * y[1] - m2 * g * y[0] * np.cos(y[1])
    b2 = y[0] * y[3] ** 2 - 2 * (c / m2) * y[0] - g * np.sin(y[1])

    dy[2] = (b1 * a22 - b2 * a12) / (a11 * a22 - a12 * a21)
    dy[3] = (b2 * a11 - b1 * a21) / (a11 * a22 - a12 * a21)

    return dy

t_fin = 20
t = np.linspace(0, t_fin, 1001)

'''
    Начальные параметры системы
'''
x0 = 4
m1 = 1 # Масса обруча
m2 = 0.5 # Масса грузика
R = 1 # Радиус обруча
c = 5 # Жесткость пружин, для грузика
c1 = 5 # Жесткости пружины для обруча
g = 9.81 # Ускорение свободного падения
phi0 = pi/2 # Угол
dphi0 = 1 # Угловая скорость
s0 = 0 # Начальное положение грузика
ds0 = 0 # Начальная скорость грузика
y0 = [s0, phi0, ds0, dphi0] # Вектор начального состояния системы. Это все величины, которые могут менять при движении
'''
    Мы получаем состояния системы в различные промежутки времени
'''
Y = odeint(fnc, y0, t, (m1, m2, c, c1, R, g)) # Решение дифференциального уравнения

s = Y[:, 0]

ds = Y[:, 2]
dds = [fnc(y, time, m1, m2, c, c1, R, g)[2] for y, time in zip(Y, t)]

phi = Y[:, 1]
dphi = Y[:, 3]
ddphi = [fnc(y, time, m1, m2, c, c1, R, g)[3] for y, time in zip(Y, t)]

angles = np.linspace(0, 2 * pi, 360)

box_w = 0.4
box_h = 0.2
def spring(k, h, w):
    x = np.linspace(0, h, 100)
    return np.array([x, np.sin(2 * math.pi / (h / k) * x) * w])

box_x_tmp = np.array([-box_h / 2, -box_h / 2, box_h / 2, box_h / 2, -box_h / 2])
box_y_tmp = np.array([-box_w / 2, box_w / 2, box_w / 2, -box_w / 2, -box_w / 2])

F_friction = np.zeros(len(t))
N = np.zeros(len(t))

ring_dots_x = np.zeros([len(t), len(angles)])
ring_dots_y = np.zeros([len(t), len(angles)])

box_dots_x = np.zeros([len(t), 5])
box_dots_y = np.zeros([len(t), 5])

spring_a_x = np.zeros([len(t), 100])
spring_a_y = np.zeros([len(t), 100])
spring_b_x = np.zeros([len(t), 100])
spring_b_y = np.zeros([len(t), 100])

spring_c_x = np.zeros([len(t), 100])
spring_c_y = np.zeros([len(t), 100])

for i in range(len(t)):
    F_friction[i] = (m1 + m2) * R * ddphi[i] - m2 * (dds[i] - s[i] * dphi[i]**2) * np.cos(phi[i]) + m2*(2*ds[i]*dphi[i] + s[i]*ddphi[i]) * np.sin(phi[i]) + c1*R*phi[i]

    N[i] = m2*((dds[i] - s[i]*(dphi[i]**2))*np.sin(phi[i])+(2*ds[i]*dphi[i] + s[i]*ddphi[i])*np.cos(phi[i])) + (m1+m2)*g

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


'''
    Графики
'''
fig_for_graphs = plt.figure(figsize=[13, 7])

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 1)
ax_for_graphs.plot(t, F_friction, color='black')
ax_for_graphs.set_title("F(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 2)
ax_for_graphs.plot(t, N, color='black')
ax_for_graphs.set_title("N(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 3)
ax_for_graphs.plot(t, s, color='blue')
ax_for_graphs.set_title("s(t)")
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

ax_for_graphs = fig_for_graphs.add_subplot(2, 2, 4)
ax_for_graphs.plot(t, phi, color='red')
ax_for_graphs.set_title('phi(t)')
ax_for_graphs.set(xlim=[0, t_fin])
ax_for_graphs.grid(True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.axis("equal")

surface = ax.plot([0, 0, 8], [5, 0, 0], "black")
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