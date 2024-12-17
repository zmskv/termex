import numpy as np
import sympy as sp
from numba import njit
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
class Particle:
    def __init__(self, r_expr, phi_expr, t):
        self.t = t
        self.r_expr = r_expr
        self.phi_expr = phi_expr
        self.r_func = sp.lambdify(t, r_expr, 'numpy')
        self.phi_func = sp.lambdify(t, phi_expr, 'numpy')
        self.r_dot_expr = sp.diff(r_expr, t)
        self.phi_dot_expr = sp.diff(phi_expr, t)
        self.r_dot_func = sp.lambdify(t, self.r_dot_expr, 'numpy')
        self.phi_dot_func = sp.lambdify(t, self.phi_dot_expr, 'numpy')

    @staticmethod
    @njit
    def compute_position(r, phi):
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        return x, y
    
    @staticmethod
    @njit
    def compute_velocity(r, phi, r_dot, phi_dot):
        vx = r_dot * np.cos(phi) - r * phi_dot * np.sin(phi)
        vy = r_dot * np.sin(phi) + r * phi_dot * np.cos(phi)
        return vx, vy
    
    @staticmethod
    @njit
    def compute_acceleration(r, phi, r_dot, phi_dot, r_ddot, phi_ddot):
        ax = (r_ddot - r * phi_dot**2) * np.cos(phi) - (2 * r_dot * phi_dot + r * phi_ddot) * np.sin(phi)
        ay = (r_ddot - r * phi_dot**2) * np.sin(phi) + (2 * r_dot * phi_dot + r * phi_ddot) * np.cos(phi)
        return ax, ay
    
    def get_positions(self, T):
        r = self.r_func(T)
        phi = self.phi_func(T)
        x, y = self.compute_position(r, phi)
        return x, y
    
    def get_velocities(self, T):
        r = self.r_func(T)
        phi = self.phi_func(T)
        r_dot = self.r_dot_func(T)
        phi_dot = self.phi_dot_func(T)
        vx, vy = self.compute_velocity(r, phi, r_dot, phi_dot)
        return vx, vy
    
    def get_accelerations(self, T):
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
        self.ax = ax
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.color = color
        self.scale = scale
        self.arrow = self.ax.quiver(self.x, self.y, self.dx * self.scale, self.dy * self.scale, color=self.color, angles='xy', scale_units='xy', scale=1)
    def update(self, x, y, dx, dy):
        self.arrow.remove()
        self.arrow = self.ax.quiver(x, y, dx * self.scale, dy * self.scale, color=self.color, angles='xy', scale_units='xy', scale=1)


class Plotter:
    def __init__(self, ax):
        self.ax = ax
        self.line, = ax.plot([], [], 'o', lw=2)
        self.trace, = ax.plot([], [], '-', lw=1)
        self.velocity_arrow = Arrow(ax, 0, 0, 0, 0, color='r', scale=0.1)
        self.acceleration_arrow = Arrow(ax, 0, 0, 0, 0, color='g', scale=0.01)
        self.legend_elements = [
            plt.Line2D([0], [0], color='r', lw=2, label='Velocity'),
            plt.Line2D([0], [0], color='g', lw=2, label='Acceleration')
        ]
        self.ax.legend(handles=self.legend_elements, loc='upper left')
    def init(self):
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.velocity_arrow.update(0, 0, 0, 0)
        self.acceleration_arrow.update(0, 0, 0, 0)
        return self.line, self.trace, self.velocity_arrow.arrow, self.acceleration_arrow.arrow
    def update(self, x, y, trace_x, trace_y, vx, vy, ax, ay):
        self.line.set_data([x], [y])
        self.trace.set_data(trace_x, trace_y)
        self.velocity_arrow.update(x, y, vx, vy)
        self.acceleration_arrow.update(x, y, ax, ay)
        return self.line, self.trace, self.velocity_arrow.arrow, self.acceleration_arrow.arrow
    def reset_plot(self):
        self.line.set_data([], [])
        self.trace.set_data([], [])
        self.velocity_arrow.update(0, 0, 0, 0)
        self.acceleration_arrow.update(0, 0, 0, 0)


class Animator:
    def __init__(self, particle, plotter, T):
        self.particle = particle
        self.plotter = plotter
        self.T = T
        self.x, self.y = particle.get_positions(T)
        self.vx, self.vy = particle.get_velocities(T)
        self.ax, self.ay = particle.get_accelerations(T)
        self.trace_x, self.trace_y = [], []
    def init(self):
        return self.plotter.init()
    def animate(self, i):
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
        ani = animation.FuncAnimation(
            self.plotter.ax.figure, self.animate, init_func=self.init,
            frames=len(self.T), interval=interval, blit=True
        )
        plt.show(block=False)  # Не блокировать выполнение после показа графика
        plt.pause(len(self.T) * interval / 1000 + 2)  # Пауза на время анимации + 2 секунды
        plt.close()  # Закрыть окно с графиком
        self.plotter.reset_plot()  # Сбросить график


if __name__ == "__main__":
    t = sp.Symbol('t')
    r_expr = 2 + sp.sin(12 * t)
    phi_expr = t + 0.2 * sp.cos(13  * t)
    T = np.linspace(1, 10, 1000)
    particle = Particle(r_expr, phi_expr, t)
    fig, ax = plt.subplots()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    plotter = Plotter(ax)
    animator = Animator(particle, plotter, T)
    animator.start()