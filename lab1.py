import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import jit


@jit(nopython=True)
def r(t):
    return 1 - np.sin(t)


@jit(nopython=True)
def phi(t):
    return 5 * t


@jit(nopython=True)
def angular_velocity(t):
    return 5  # Угловая скорость постоянна


@jit(nopython=True)
def tangential_velocity(t):
    return r(t) * angular_velocity(t)


@jit(nopython=True)
def radial_velocity(t):
    return np.cos(t)  # Радиальная составляющая скорости


@jit(nopython=True)
def radial_acceleration(t):
    return -np.cos(t) * angular_velocity(t)


@jit(nopython=True)
def normal_acceleration(t):
    return tangential_velocity(t) ** 2 / r(t)


@jit(nopython=True)
def x(t):
    return r(t) * np.cos(phi(t))


@jit(nopython=True)
def y(t):
    return r(t) * np.sin(phi(t))


@jit(nopython=True)
def curvature_radius(t):
    dx_dt = -r(t) * np.sin(phi(t)) * angular_velocity(t)
    dy_dt = r(t) * np.cos(phi(t)) * angular_velocity(t)
    d2x_dt2 = -r(t) * np.cos(phi(t)) * angular_velocity(t) ** 2
    d2y_dt2 = -r(t) * np.sin(phi(t)) * angular_velocity(t) ** 2
    num = (dx_dt**2 + dy_dt**2) ** 1.5
    den = np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt)
    return num / den if den != 0 else np.inf


class Arrow:
    def __init__(self, ax, color="black", scale=1.0):
        self.ax = ax
        self.color = color
        self.scale = scale
        (self.line,) = ax.plot([], [], color=color, lw=1.5)
        self.head = ax.plot([], [], color=color, marker=(3, 0, 0), markersize=10)[
            0
        ]  # Треугольник

    def update(self, start_x, start_y, end_x, end_y):
        # Обновляем координаты линии
        self.line.set_data([start_x, end_x], [start_y, end_y])

        # Корректируем угол поворота для треугольника
        angle = (
            np.arctan2(end_y - start_y, end_x - start_x) * 180 / np.pi - 90
        )  # Корректировка угла поворота
        self.head.set_marker((3, 0, angle))  # Поворот треугольника
        self.head.set_data([end_x], [end_y])


class MotionVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect("equal")
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_title("Траектория движения точки")
        self.ax.set_xlabel("X координата")
        self.ax.set_ylabel("Y координата")
        self.ax.grid(True)
        self.ax.axhline(0, color="black", linewidth=0.5)
        self.ax.axvline(0, color="black", linewidth=0.5)

        # Объекты для анимации
        (self.point,) = self.ax.plot([], [], "bo", markersize=8)
        (self.trajectory,) = self.ax.plot([], [], "b-", lw=1)

        # Параметры для хранения координат траектории
        self.t_vals = np.linspace(0, 2 * np.pi, 200)
        self.x_vals = np.array([x(t) for t in self.t_vals])
        self.y_vals = np.array([y(t) for t in self.t_vals])
        self.v_vals = np.array([tangential_velocity(t) for t in self.t_vals])
        self.a_vals = np.array([radial_acceleration(t) for t in self.t_vals])
        self.curvature_radii = np.array([curvature_radius(t) for t in self.t_vals])

        # Стрелки скорости, радиус-вектора и ускорения
        self.velocity_arrow = Arrow(self.ax, color="green")
        self.radius_vector_arrow = Arrow(self.ax, color="blue")
        self.acceleration_arrow = Arrow(self.ax, color="black")

        # Текущая дуга радиуса кривизны
        self.curvature_circle = None

        # Добавляем списки для стрелок и окружностей
        self.arrows = []  # Список для стрелок
        self.circles = []  # Список для дуг радиуса кривизны

        # Текстовые поля для отображения величин
        self.angular_text = self.ax.text(
            0.05, 0.95, "", transform=self.ax.transAxes, fontsize=10, color="red"
        )
        self.velocity_text = self.ax.text(
            0.05, 0.90, "", transform=self.ax.transAxes, fontsize=10, color="green"
        )
        self.acceleration_text = self.ax.text(
            0.05, 0.85, "", transform=self.ax.transAxes, fontsize=10, color="black"
        )
        self.curvature_radius_text = self.ax.text(
            0.05, 0.80, "", transform=self.ax.transAxes, fontsize=10, color="purple"
        )

        # Добавляем текстовые метки для векторов внизу графика
        self.vector_labels = self.ax.text(
            0.05, -0.1, "", transform=self.ax.transAxes, fontsize=10, color="black"
        )

    def init_animation(self):
        self.point.set_data([], [])
        self.trajectory.set_data([], [])
        return self.point, self.trajectory

    def draw_arrow(
        self, start_x, start_y, end_x, end_y, color, scale=1.0, head_scale=1.0
    ):
        # Создаем линию стрелки
        (line,) = self.ax.plot([start_x, end_x], [start_y, end_y], color=color, lw=2)
        self.arrows.append(line)

        # Добавляем треугольник в конце стрелки
        angle = np.arctan2(end_y - start_y, end_x - start_x) * 180 / np.pi - 90
        head = self.ax.plot(
            [end_x],
            [end_y],
            color=color,
            marker=(3, 0, angle),
            markersize=10 * head_scale,
        )[0]
        self.arrows.append(head)

    def animate(self, frame_idx):
        # Вычисление координат точки и скоростей
        px, py = self.x_vals[frame_idx], self.y_vals[frame_idx]
        tangential_speed = self.v_vals[frame_idx]
        acc_val = self.a_vals[frame_idx]
        curvature_radius_val = self.curvature_radii[frame_idx]

        # Обновляем положение точки и траектории
        self.point.set_data([px], [py])
        self.trajectory.set_data(
            self.x_vals[: frame_idx + 1], self.y_vals[: frame_idx + 1]
        )

        # Очищаем стрелки и окружности
        for arrow in self.arrows:
            arrow.remove()
        self.arrows.clear()

        for circle in self.circles:
            circle.remove()
        self.circles.clear()

        # Радиус-вектор — стрелка от центра
        scale = 0.3  # Масштаб для стрелок
        head_scale = 1.0  # Масштаб для наконечника стрелки
        self.draw_arrow(0, 0, px, py, "green", scale=scale, head_scale=head_scale)

        # Расчет вектора касательной (скорость)
        vx, vy = -r(self.t_vals[frame_idx]) * np.sin(phi(self.t_vals[frame_idx])), r(
            self.t_vals[frame_idx]
        ) * np.cos(phi(self.t_vals[frame_idx]))

        # Нормальный вектор
        normal_vx = vy / np.sqrt(vx**2 + vy**2)
        normal_vy = -vx / np.sqrt(vx**2 + vy**2)

        # Центр кривизны
        cx = px - normal_vx * curvature_radius_val
        cy = py - normal_vy * curvature_radius_val

        # Отрисовка радиуса кривизны в виде дуги
        curvature_arc = plt.Circle(
            (cx, cy), curvature_radius_val, color="pink", fill=False, linestyle="--"
        )
        self.ax.add_artist(curvature_arc)
        self.circles.append(curvature_arc)

        # Ускорение — стрелка, отображающая изменение скорости
        self.draw_arrow(
            px,
            py,
            px + vx / np.sqrt(vx**2 + vy**2) * acc_val * 0.5,
            py + vy / np.sqrt(vx**2 + vy**2) * acc_val * 0.5,
            "black",
            scale=scale,
            head_scale=head_scale,
        )

        # Общая скорость — векторная сумма тангенциальной и нормальной скоростей
        normal_speed = tangential_speed / curvature_radius_val
        total_speed = np.sqrt(tangential_speed**2 + normal_speed**2)
        total_vx = vx / np.sqrt(vx**2 + vy**2) * total_speed
        total_vy = vy / np.sqrt(vx**2 + vy**2) * total_speed
        self.draw_arrow(
            px,
            py,
            px + total_vx * 0.3,
            py + total_vy * 0.3,
            "blue",
            scale=scale,
            head_scale=head_scale,
        )

        # Обновляем значения скоростей и ускорения
        self.angular_text.set_text(
            f"Angular velocity: {angular_velocity(self.t_vals[frame_idx]):.2f} rad/s"
        )
        self.acceleration_text.set_text(f"Acceleration: {acc_val:.2f} units/s²")
        self.velocity_text.set_text(f"Total velocity: {total_speed:.2f} units/s")
        self.curvature_radius_text.set_text(
            f"Curvature radius: {curvature_radius_val:.2f} units"
        )

        # Обновляем текстовые метки для векторов внизу графика
        self.vector_labels.set_text(
            "Blue: Radius Vector, Green: Velocity, Black: Acceleration"
        )

        return (
            self.point,
            self.trajectory,
            self.angular_text,
            self.acceleration_text,
            self.velocity_text,
            self.curvature_radius_text,
            self.vector_labels,
        )

    def start_animation(self):
        ani = FuncAnimation(
            self.fig,
            self.animate,
            frames=len(self.t_vals),
            init_func=self.init_animation,
            interval=50,
        )
        plt.show()


# Запуск визуализации
visualizer = MotionVisualizer()
visualizer.start_animation()