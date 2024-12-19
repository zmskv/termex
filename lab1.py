import numpy as np
import sympy as sp
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time

class Particle:
    def __init__(self, r_expr, phi_expr, t):
        """
        Инициализация объекта Particle.

        :param r_expr: Символьное выражение для радиуса r(t).
        :param phi_expr: Символьное выражение для угла phi(t).
        :param t: Символьная переменная времени.
        """
        self.t = t
        self.r_expr = r_expr
        self.phi_expr = phi_expr
        self.r_func = sp.lambdify(t, r_expr, 'numpy')  # Функция для вычисления r(t)
        self.phi_func = sp.lambdify(t, phi_expr, 'numpy')  # Функция для вычисления phi(t)
        self.r_dot_expr = sp.diff(r_expr, t)  # Производная r по времени
        self.phi_dot_expr = sp.diff(phi_expr, t)  # Производная phi по времени
        self.r_dot_func = sp.lambdify(t, self.r_dot_expr, 'numpy')  # Функция для вычисления dr/dt
        self.phi_dot_func = sp.lambdify(t, self.phi_dot_expr, 'numpy')  # Функция для вычисления dphi/dt

    @staticmethod
    @njit
    def compute_position(r, phi):
        """
        Вычисляет декартовы координаты (x, y) из полярных (r, phi).

        :param r: Массив радиусов.
        :param phi: Массив углов.
        :return: Кортеж (x, y) декартовых координат.
        """
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y
    
    @staticmethod
    @njit
    def compute_velocity(r, phi, r_dot, phi_dot):
        """
        Вычисляет скорость (vx, vy) в декартовых координатах.

        :param r: Массив радиусов.
        :param phi: Массив углов.
        :param r_dot: Массив производных радиуса по времени.
        :param phi_dot: Массив производных угла по времени.
        :return: Кортеж (vx, vy) компонентов скорости.
        """
        vx = r_dot * np.cos(phi) - r * phi_dot * np.sin(phi)
        vy = r_dot * np.sin(phi) + r * phi_dot * np.cos(phi)
        return vx, vy
    
    @staticmethod
    @njit
    def compute_acceleration(r, phi, r_dot, phi_dot, r_ddot, phi_ddot):
        """
        Вычисляет ускорени (ax, ay) в декартовых координатах.

        :param r: Массив радиусов.
        :param phi: Массив углов.
        :param r_dot: Массив производных радиуса по времени.
        :param phi_dot: Массив производных угла по времени.
        :param r_ddot: Массив вторых производных радиуса по времени.
        :param phi_ddot: Массив вторых производных угла по времени.
        :return: Кортеж (ax, ay) компонентов ускорения.
        """
        ax = (r_ddot - r * phi_dot**2) * np.cos(phi) - (2 * r_dot * phi_dot + r * phi_ddot) * np.sin(phi)
        ay = (r_ddot - r * phi_dot**2) * np.sin(phi) + (2 * r_dot * phi_dot + r * phi_ddot) * np.cos(phi)
        return ax, ay
    
    def get_positions(self, T):
        """
        Вычисляет координаты (x, y) для заданных моментов времени T.

        :param T: Массив моментов времени.
        :return: Кортеж (x, y) координат.
        """
        r = self.r_func(T)
        phi = self.phi_func(T)
        x, y = self.compute_position(r, phi)
        return x, y
    
    def get_velocities(self, T):
        """
        Вычисляет скорость (vx, vy) для заданных моментов времени T.

        :param T: Массив моментов времени.
        :return: Кортеж (vx, vy) скоростей.
        """
        r = self.r_func(T)
        phi = self.phi_func(T)
        r_dot = self.r_dot_func(T)
        phi_dot = self.phi_dot_func(T)
        vx, vy = self.compute_velocity(r, phi, r_dot, phi_dot)
        return vx, vy
    
    def get_accelerations(self, T):
        """
        Вычисляет ускорение (ax, ay) для заданных моментов времени T.

        :param T: Массив моментов времени.
        :return: Кортеж (ax, ay) ускорений.
        """
        r = self.r_func(T)
        phi = self.phi_func(T)
        r_dot = self.r_dot_func(T)
        phi_dot = self.phi_dot_func(T)
        r_ddot_expr = sp.diff(self.r_dot_expr, t)
        phi_ddot_expr = sp.diff(self.phi_dot_expr, t)
        r_ddot_func = sp.lambdify(t, r_ddot_expr, 'numpy')
        phi_ddot_func = sp.lambdify(t, phi_ddot_expr, 'numpy')
        r_ddot = r_ddot_func(T)
        phi_ddot = phi_ddot_func(T)
        ax, ay = self.compute_acceleration(r, phi, r_dot, phi_dot, r_ddot, phi_ddot)
        return ax, ay

class Arrow:
    def __init__(self, ax, x, y, dx, dy, color='k', scale=1.0):
        """
        Инициализация объекта Arrow.

        :param ax: Оси, на которых рисуется стрелка.
        :param x: Начальная координата x.
        :param y: Начальная координата y.
        :param dx: Смещение по оси x.
        :param dy: Смещение по оси y.
        :param color: Цвет стрелки.
        :param scale: Масштаб стрелки.
        """
        self.ax = ax
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.scale = scale
        self.arrow = self.ax.quiver(self.x, self.y, self.dx * self.scale, self.dy * self.scale, color=self.color, angles='xy', scale_units='xy', scale=1)

    def update(self, x, y, dx, dy):
        """
        Обновляет положение и направление стрелки.

        :param x: Новая начальная координата x.
        :param y: Новая начальная координата y.
        :param dx: Новое смещение по оси x.
        :param dy: Новое смещение по оси y.
        """
        self.arrow.remove()
        self.arrow = self.ax.quiver(x, y, dx * self.scale, dy * self.scale, color=self.color, angles='xy', scale_units='xy', scale=1)

class Plotter:
    def __init__(self, ax):
        """
        Инициализация объекта Plotter.

        :param ax: Оси, на которых рисуются графики.
        """
        self.ax = ax
        self.line, = ax.plot([], [], 'o', lw=2)  # Точка текущего положения
        self.trace, = ax.plot([], [], '-', lw=1)  # Траектория движения
        self.velocity_arrow = Arrow(ax, 0, 0, 0, 0, color='r', scale=0.1)  # Стрелка скорости
        self.acceleration_arrow = Arrow(ax, 0, 0, 0, 0, color='g', scale=0.01)  # Стрелка ускорения
        self.legend_elements = [
            plt.Line2D([0], [0], color='r', lw=2, label='Velocity'),
            plt.Line2D([0], [0], color='g', lw=2, label='Acceleration')
        ]
        self.ax.legend(handles=self.legend_elements, loc='upper left')  # Легенда

    def init(self):
        """
        Инициализация графиков.

        :return: Инициализированные объекты для анимации.
        """
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.velocity_arrow.update(0, 0, 0, 0)
        self.acceleration_arrow.update(0, 0, 0, 0)
        return self.line, self.trace, self.velocity_arrow.arrow, self.acceleration_arrow.arrow

    def update(self, x, y, trace_x, trace_y, vx, vy, ax, ay):
        """
        Обновляет графики для текущего кадра.

        :param x: Текущая координата x.
        :param y: Текущая координата y.
        :param trace_x: Список координат x траектории.
        :param trace_y: Список координат y траектории.
        :param vx: Скорость по оси x.
        :param vy: Скорость по оси y.
        :param ax: Ускорениe по оси x.
        :param ay: Ускорение по оси y.
        :return: Обновленные объекты для анимации.
        """
        self.line.set_data([x], [y])
        self.trace.set_data(trace_x, trace_y)
        self.velocity_arrow.update(x, y, vx, vy)
        self.acceleration_arrow.update(x, y, ax, ay)
        return self.line, self.trace, self.velocity_arrow.arrow, self.acceleration_arrow.arrow
    
    def reset_plot(self):
        """
        Сбрасывает график.

        """
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.velocity_arrow.update(0, 0, 0, 0)
        self.acceleration_arrow.update(0, 0, 0, 0)

class Animator:
    def __init__(self, particle, plotter, T):
        """
        Инициализация объекта Animator.

        :param particle: Объект Particle.
        :param plotter: Объект Plotter.
        :param T: Массив моментов времени.
        """
        self.particle = particle
        self.plotter = plotter
        self.T = T
        self.x, self.y = particle.get_positions(T)
        self.vx, self.vy = particle.get_velocities(T)
        self.ax, self.ay = particle.get_accelerations(T)
        self.trace_x, self.trace_y = [], []

    def init(self):
        """
        Инициализация анимации.

        :return: Инициализированные объекты для анимации.
        """
        return self.plotter.init()

    def animate(self, i):
        """
        Обновляет графики для текущего кадра.

        :param i: Индекс текущего кадра.
        :return: Обновленные объекты для анимации.
        """
        x = self.x[i]
        y = self.y[i]
        vx = self.vx[i]
        vy = self.vy[i]
        ax = self.ax[i]
        ay = self.ay[i]
        self.trace_x.append(x)
        self.trace_y.append(y)
        return self.plotter.update(x, y, self.trace_x, self.trace_y, vx, vy, ax, ay)

    def start(self, interval=20):
        """
        Запускает анимацию.

        :param interval: Интервал между кадрами в миллисекундах.
        """
        ani = animation.FuncAnimation(
            self.plotter.ax.figure, self.animate, init_func=self.init,
            frames=len(self.T), interval=interval, blit=True
        )
        plt.show(block=False) 
        plt.pause(len(self.T) * interval / 1000 + 2)  
        plt.close()  
        self.plotter.reset_plot()  


if __name__ == "__main__":
    t = sp.Symbol('t')
    r_expr = 2 + sp.sin(12 * t)  # Задаем радиус как функцию от времени
    phi_expr = t + 0.2 * sp.cos(13 * t)  # Задаем угол как функцию от времени
    T = np.linspace(1, 10, 1000)  # Диапазон времени
    particle = Particle(r_expr, phi_expr, t)  # Создаем объект Particle
    fig, ax = plt.subplots()  # Создаем фигуру и оси
    ax.set_xlim(-3, 3)  
    ax.set_ylim(-3, 3)  
    ax.set_aspect('equal') 
    plotter = Plotter(ax)  # Создаем объект Plotter
    animator = Animator(particle, plotter, T)  # Создаем объект Animator
    animator.start()  # Запускаем анимацию