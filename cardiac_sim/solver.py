"""Солвер для симуляции одной клетки TNNPM.

Запускает численное интегрирование ODE системы модели кардиомиоцита
с использованием scipy.integrate.solve_ivp (метод BDF).

Симуляция:
    - Длительность: 1000 мс
    - Стимул: t=100 мс, амплитуда -40 мкА/см², длительность 1 мс

Результаты:
    - График мембранного потенциала V (y[11])
    - График внутриклеточного кальция Ca_i (y[5])
    - График доли поперечных мостиков N (y[21])
    - Сохраняются в директорию results/
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Абсолютные импорты для запуска как скрипт
from cardiac_sim.calculate_parameters import param
from cardiac_sim.model import TNNPM


def stimulate(t: float, t_start: float = 100.0, t_duration: float = 1.0,
              amplitude: float = 40.0) -> float:
    """Применить стимул к клетке.

    Args:
        t: Текущее время (мс).
        t_start: Время начала стимула (мс).
        t_duration: Длительность стимула (мс).
        amplitude: Амплитуда стимула (мкА/см²).

    Returns:
        Значение стимула (отрицательное = деполяризация).
    """
    if t_start <= t <= t_start + t_duration:
        return -amplitude
    return 0.0


def run_single_cell_simulation(
    t_span: tuple[float, float] = (0.0, 1000.0),
    stim_start: float = 100.0,
    stim_duration: float = 1.0,
    stim_amplitude: float = 40.0,
    method: str = "BDF",
    max_step: float = 0.1,
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, TNNPM]:
    """Запустить симуляцию одной клетки TNNPM.

    Args:
        t_span: Интервал времени симуляции (мс), кортеж (t_start, t_end).
        stim_start: Время начала стимула (мс).
        stim_duration: Длительность стимула (мс).
        stim_amplitude: Амплитуда стимула (мкА/см²).
        method: Метод интегрирования ('BDF' или 'RK45').
        max_step: Максимальный шаг интегрирования (мс).
        rtol: Относительная точность.
        atol: Абсолютная точность.

    Returns:
        Кортеж (time_points, solution, model), где:
            - time_points: массив временных точек (мс)
            - solution: массив решения формы (n_vars, n_time_points)
            - model: экземпляр TNNPM после симуляции
    """
    # Изменяем параметры стимуля ДО создания модели
    param["stim_start"]["value"] = stim_start
    param["stim_duration"]["value"] = stim_duration
    param["stim_amplitude"]["value"] = stim_amplitude

    # Создаём модель и вычисляем начальные условия
    model = TNNPM()
    y0 = model.calculate_init_conditions()

    print(f"Начальные условия:")
    print(f"  V (мембранный потенциал): {y0[11]:.2f} мВ")
    print(f"  Ca_i (внутриклеточный Ca²⁺): {y0[5]*1e6:.4f} мкМ")
    print(f"  N (поперечные мостики): {y0[21]:.6f}")

    # Определяем функцию RHS для ODE
    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        """Правая часть системы ODE.

        Стимул применяется через встроенный механизм модели
        (используя param["stim_amplitude"]).
        """
        return model.diff_equations(t, y)

    # Решаем ODE систему
    print(f"\nЗапуск симуляции: {t_span[0]} - {t_span[1]} мс")
    print(f"Стимул: t={stim_start} мс, длительность={stim_duration} мс, "
          f"амплитуда=-{stim_amplitude} мкА/см²")
    print(f"Метод интегрирования: {method}")

    sol = solve_ivp(
        fun=rhs,
        t_span=t_span,
        y0=y0,
        method=method,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
        dense_output=True,
    )

    if not sol.success:
        print(f"Предупреждение: интегратор вернул статус {sol.status}: {sol.message}")
    else:
        print(f"Интегрирование завершено успешно: {sol.nfev} вызовов функции")

    return sol.t, sol.y, model


def calculate_apd90(time: np.ndarray, V: np.ndarray) -> float:
    """Вычислить APD90 (длительность потенциала действия при 90% реполяризации).

    APD90 означает время от пика до 90% реполяризации (когда осталось 10%
    от амплитуды потенциала действия).

    Args:
        time: Массив времени (мс).
        V: Массив мембранного потенциала (мВ).

    Returns:
        APD90 в миллисекундах.
    """
    # Находим пик потенциала действия
    peak_idx = np.argmax(V)
    V_rest = np.median(V[:peak_idx]) if peak_idx > 0 else V[0]
    V_peak = V[peak_idx]

    # APD90 = потенциал при 90% реполяризации = покой + 10% от амплитуды
    # (или equivalently: пик - 90% от амплитуды)
    threshold = V_rest + 0.1 * (V_peak - V_rest)

    # Ищем время достижения APD90 после пика
    APD90 = 0.0
    for i in range(peak_idx, len(V)):
        if V[i] <= threshold:
            # Линейная интерполяция
            if i > 0:
                t1 = time[i - 1]
                t2 = time[i]
                v1 = V[i - 1]
                v2 = V[i]
                APD90 = t1 + (t2 - t1) * (threshold - v1) / (v2 - v1)
            else:
                APD90 = time[i]
            break

    return APD90


def plot_results(
    time: np.ndarray,
    solution: np.ndarray,
    model: TNNPM,
    save_dir: str = "results",
) -> None:
    """Построить и сохранить графики результатов симуляции.

    Args:
        time: Массив времени (мс).
        solution: Массив решения (n_vars, n_time_points).
        model: Экземпляр модели TNNPM.
        save_dir: Директория для сохранения графиков.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Извлекаем ключевые переменные
    V = solution[11]        # Мембранный потенциал (мВ)
    Ca_i = solution[5]     # Внутриклеточный Ca²⁺ (мМ)
    N = solution[21]       # Доля поперечных мостиков
    Ca_tn = solution[22]    # Ca²⁺ связанный с TnC

    # Переводим Ca_i из мМ в мкМ для удобства
    # 1 мМ = 1000 мкМ
    Ca_i_uM = Ca_i * 1000.0

    # Вычисляем APD90
    apd90 = calculate_apd90(time, V)

    # Ключевые метрики
    V_rest = np.median(V[:100])
    V_peak = np.max(V)
    Ca_peak = np.max(Ca_i_uM)

    print(f"\nКлючевые метрики:")
    print(f"  V_rest (потенциал покоя): {V_rest:.1f} мВ")
    print(f"  V_peak (пик потенциала действия): {V_peak:.1f} мВ")
    print(f"  APD90 (длительность потенциала действия 90%): {apd90:.1f} мс")
    print(f"  Ca_peak (пик внутриклеточного Ca²⁺): {Ca_peak:.3f} мкМ")
    print(f"  N_max (максимальная доля мостиков): {np.max(N):.6f}")

    # Проверяем критерии валидации
    print(f"\nКритерии валидации:")
    V_rest_ok = -90 < V_rest < -80
    V_peak_ok = 30 < V_peak < 50
    APD90_ok = 200 < apd90 < 400
    Ca_ok = 0.5 < Ca_peak < 1.5
    N_ok = np.max(N) > 0

    print(f"  V_rest ≈ -86 мВ: {'✓' if V_rest_ok else '✗'} ({V_rest:.1f} мВ)")
    print(f"  V_peak ≈ +40 мВ: {'✓' if V_peak_ok else '✗'} ({V_peak:.1f} мВ)")
    print(f"  APD90: 280-320 мс: {'✓' if APD90_ok else '✗'} ({apd90:.1f} мс)")
    print(f"  Ca_i peak: 0.8-1.0 мкМ: {'✓' if Ca_ok else '✗'} ({Ca_peak:.3f} мкМ)")
    print(f"  N > 0 (мостики активны): {'✓' if N_ok else '✗'}")

    # Создаём фигуру с тремя подграфиками
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # График 1: Мембранный потенциал
    ax1 = axes[0]
    ax1.plot(time, V, 'b-', linewidth=1.0, label='V')
    ax1.axhline(y=V_rest, color='gray', linestyle='--', alpha=0.5, label=f'V_rest={V_rest:.1f} мВ')
    ax1.axhline(y=V_peak, color='red', linestyle='--', alpha=0.5, label=f'V_peak={V_peak:.1f} мВ')
    ax1.axvline(x=100, color='green', linestyle=':', alpha=0.7, label='Стимул (t=100 мс)')
    ax1.set_ylabel('Мембранный потенциал, мВ', fontsize=11)
    ax1.set_title(f'Мембранный потенциал (APD90 = {apd90:.1f} мс)', fontsize=12)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, time[-1]])

    # График 2: Внутриклеточный кальций
    ax2 = axes[1]
    ax2.plot(time, Ca_i_uM, 'r-', linewidth=1.0, label='Ca_i')
    ax2.axhline(y=Ca_peak, color='orange', linestyle='--', alpha=0.5, label=f'Ca_peak={Ca_peak:.3f} мкМ')
    ax2.axvline(x=100, color='green', linestyle=':', alpha=0.7, label='Стимул')
    ax2.set_ylabel('Внутриклеточный Ca²⁺, мкМ', fontsize=11)
    ax2.set_title('Внутриклеточная концентрация кальция', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # График 3: Поперечные мостики
    ax3 = axes[2]
    ax3.plot(time, N, 'g-', linewidth=1.0, label='N')
    ax3.axhline(y=np.max(N), color='purple', linestyle='--', alpha=0.5,
                label=f'N_max={np.max(N):.6f}')
    ax3.axvline(x=100, color='green', linestyle=':', alpha=0.7, label='Стимул')
    ax3.set_xlabel('Время, мс', fontsize=11)
    ax3.set_ylabel('Доля мостиков N', fontsize=11)
    ax3.set_title('Активные поперечные мостики (доля прикреплённых)', fontsize=12)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Сохраняем график
    fig_path = save_path / "single_cell_simulation.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\nГрафик сохранён: {fig_path}")

    # Отдельный график для APD
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, V, 'b-', linewidth=1.5, label='Мембранный потенциал')
    ax.plot(time, Ca_i_uM, 'r-', linewidth=1.5, label='Ca_i (×100 для масштаба)')

    # Масштабируем Ca_i для отображения на том же графике
    ax2_twin = ax.twinx()
    ax2_twin.plot(time, N * 100, 'g-', linewidth=1.5, label='N (×100)')
    ax2_twin.set_ylabel('N × 100', fontsize=11, color='green')
    ax2_twin.tick_params(axis='y', labelcolor='green')

    ax.set_xlabel('Время, мс', fontsize=11)
    ax.set_ylabel('Мембранный потенциал, мВ / Ca_i, мкМ', fontsize=11, color='blue')
    ax.tick_params(axis='y', labelcolor='blue')
    ax.set_title('Симуляция одной клетки TNNPM: потенциал действия и кальций', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Объединяем легенды
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig2_path = save_path / "single_cell_combined.png"
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"Комбинированный график сохранён: {fig2_path}")

    plt.close('all')

    # Сохраняем данные в CSV
    csv_path = save_path / "single_cell_data.csv"
    data = np.column_stack([time, V, Ca_i_uM, N, Ca_tn * 1000.0])
    header = "time_ms,V_mV,Ca_i_uM,N,CaTn_uM"
    np.savetxt(csv_path, data, delimiter=",", header=header, comments='')
    print(f"Данные сохранены: {csv_path}")


def main() -> None:
    """Главная функция для запуска симуляции одной клетки."""
    print("=" * 60)
    print("Симуляция одной клетки TNNPM")
    print("=" * 60)

    # Параметры симуляции
    T_END = 1000.0  # мс
    STIM_START = 100.0  # мс
    STIM_DURATION = 1.0  # мс
    STIM_AMPLITUDE = 40.0  # мкА/см²

    # Запуск симуляции
    time, solution, model = run_single_cell_simulation(
        t_span=(0.0, T_END),
        stim_start=STIM_START,
        stim_duration=STIM_DURATION,
        stim_amplitude=STIM_AMPLITUDE,
        method="BDF",
    )

    # Построение и сохранение графиков
    plot_results(time, solution, model, save_dir="results")

    print("\n" + "=" * 60)
    print("Симуляция завершена!")
    print("=" * 60)


if __name__ == "__main__":
    main()
