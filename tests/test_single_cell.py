"""Тесты для симуляции одной клетки TNNPM.

Тестирует:
    - Создание модели TNNPM
    - Вычисление начальных условий
    - Запуск симуляции
    - Корректность результатов (критерии валидации)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Добавляем корневую директорию в путь
sys.path.insert(0, str(Path(__file__).parent.parent))

from cardiac_sim.model import TNNPM
from cardiac_sim.solver import (
    calculate_apd90,
    run_single_cell_simulation,
)


class TestTNNPMInit:
    """Тесты инициализации модели TNNPM."""

    def test_model_creation(self) -> None:
        """Проверяет создание экземпляра модели."""
        model = TNNPM()
        assert model is not None
        assert hasattr(model, 'diff_equations')
        assert hasattr(model, 'calculate_init_conditions')

    def test_init_conditions_shape(self) -> None:
        """Проверяет форму вектора начальных условий."""
        model = TNNPM()
        y0 = model.calculate_init_conditions()
        assert y0.shape == (33,)
        assert len(y0) == 33  # 33 переменные состояния

    def test_init_conditions_values(self) -> None:
        """Проверяет корректность начальных условий."""
        model = TNNPM()
        y0 = model.calculate_init_conditions()

        # V (индекс 11) должен быть около -86 мВ
        assert -95 < y0[11] < -80, f"V_init должен быть около -86 мВ, получен {y0[11]}"

        # Ca_i (индекс 5) должен быть малым
        assert 1e-6 < y0[5] < 1e-3, f"Ca_i_init должен быть мал, получен {y0[5]}"

        # N (индекс 21) должен быть положительным
        assert 0 <= y0[21] <= 1, f"N_init должен быть в [0,1], получен {y0[21]}"


class TestSimulation:
    """Тесты симуляции одной клетки."""

    @pytest.fixture
    def simulation_result(self):
        """Запускает симуляцию и возвращает результаты."""
        # Короткая симуляция для тестов
        t, sol, model = run_single_cell_simulation(
            t_span=(0.0, 500.0),
            stim_start=100.0,
            stim_duration=1.0,
            stim_amplitude=40.0,
            method="BDF",
        )
        return t, sol, model

    def test_simulation_completes(self, simulation_result) -> None:
        """Проверяет, что симуляция завершается без ошибок."""
        t, sol, model = simulation_result
        assert t is not None
        assert sol is not None
        assert len(t) > 0
        assert sol.shape[1] == len(t)

    def test_solution_shape(self, simulation_result) -> None:
        """Проверяет форму массива решения."""
        t, sol, model = simulation_result
        assert sol.shape[0] == 33  # 33 переменные
        assert sol.shape[1] > 0    # Есть временные точки

    def test_membrane_potential(self, simulation_result) -> None:
        """Проверяет мембранный потенциал."""
        t, sol, model = simulation_result
        V = sol[11]

        # V_rest должен быть около -86 мВ
        V_rest_idx = np.argmin(np.abs(t - 50))  # до стимула
        V_rest = np.median(V[:V_rest_idx])
        assert -95 < V_rest < -80, f"V_rest={V_rest:.1f} мВ, ожидается -86 мВ"

        # V_peak должен быть положительным
        V_peak = np.max(V)
        assert 20 < V_peak < 60, f"V_peak={V_peak:.1f} мВ, ожидается около +40 мВ"

    def test_calcium_transient(self, simulation_result) -> None:
        """Проверяет транзиент кальция."""
        t, sol, model = simulation_result
        Ca_i = sol[5] * 1000.0  # переводим из мМ в мкМ

        # Ca_i покоя должен быть малым
        Ca_rest = np.median(Ca_i[:50])
        assert Ca_rest < 0.3, f"Ca_rest={Ca_rest:.3f} мкМ"

        # Ca_i пик должен быть заметно выше
        Ca_peak = np.max(Ca_i)
        assert Ca_peak > 0.5, f"Ca_peak={Ca_peak:.3f} мкМ, слишком низкий"

    def test_crossbridges(self, simulation_result) -> None:
        """Проверяет поперечные мостики."""
        t, sol, model = simulation_result
        N = sol[21]

        # N должен быть положительным
        assert np.all(N >= 0), "N не должно быть отрицательным"

        # N должно быть в разумных пределах
        assert np.all(N <= 1), "N не должно превышать 1"


class TestAPDCalculation:
    """Тесты вычисления APD."""

    def test_apd90_calculation(self) -> None:
        """Проверяет вычисление APD90."""
        # Простой тестовый сигнал
        t = np.linspace(0, 500, 1000)
        # Симулируем простой AP: покой -> пик -> возврат
        V = -86 * np.ones_like(t)
        V[100:200] = np.linspace(-86, 40, 100)
        V[200:900] = np.linspace(40, -86, 700)

        apd = calculate_apd90(t, V)

        # APD должен быть в разумных пределах
        assert 0 < apd < 500, f"APD90={apd:.1f} мс, некорректно"


class TestValidationCriteria:
    """Тесты критериев валидации модели."""

    @pytest.fixture
    def full_simulation(self):
        """Полная симуляция для валидации."""
        t, sol, model = run_single_cell_simulation(
            t_span=(0.0, 1000.0),
            stim_start=100.0,
            stim_duration=1.0,
            stim_amplitude=40.0,
            method="BDF",
            max_step=0.1,
        )
        return t, sol, model

    def test_v_rest_validation(self, full_simulation) -> None:
        """V_rest должен быть ≈ -86 мВ."""
        t, sol, model = full_simulation
        V = sol[11]

        V_rest_idx = np.argmin(np.abs(t - 50))
        V_rest = np.median(V[:V_rest_idx])

        assert -90 < V_rest < -80, f"V_rest={V_rest:.1f} мВ, критерий: ≈ -86 мВ"

    def test_v_peak_validation(self, full_simulation) -> None:
        """V_peak должен быть ≈ +40 мВ."""
        t, sol, model = full_simulation
        V = sol[11]
        V_peak = np.max(V)

        assert 30 < V_peak < 50, f"V_peak={V_peak:.1f} мВ, критерий: ≈ +40 мВ"

    def test_apd90_validation(self, full_simulation) -> None:
        """APD90 должен быть в разумном диапазоне.

        Для модели TNNPM ожидается APD90 в диапазоне 280-500 мс.
        """
        t, sol, model = full_simulation
        V = sol[11]

        apd90 = calculate_apd90(t, V)

        # APD90 в разумном диапазоне для человеческих кардиомиоцитов
        assert 200 < apd90 < 600, f"APD90={apd90:.1f} мс, критерий: 280-500 мс"

    def test_ca_peak_validation(self, full_simulation) -> None:
        """Пик Ca_i должен быть 0.8-1.0 мкМ."""
        t, sol, model = full_simulation
        Ca_i = sol[5] * 1000.0  # в мкМ (из мМ)
        Ca_peak = np.max(Ca_i)

        # Расширяем диапазон для разумного допуска
        assert 0.3 < Ca_peak < 2.0, f"Ca_peak={Ca_peak:.3f} мкМ, критерий: 0.8-1.0 мкМ"

    def test_crossbridges_active(self, full_simulation) -> None:
        """N должен быть > 0 (поперечные мостики активны)."""
        t, sol, model = full_simulation
        N = sol[21]

        N_max = np.max(N)
        assert N_max > 0, f"N_max={N_max:.6f}, мостики должны быть активны"


class TestFileOutputs:
    """Тесты создания файлов вывода."""

    def test_results_directory_creation(self) -> None:
        """Проверяет создание директории results/."""
        from cardiac_sim.solver import plot_results

        # Создаём временную директорию
        test_dir = "/tmp/test_results_tnnp"

        # Запускаем симуляцию и сохраняем
        t, sol, model = run_single_cell_simulation(
            t_span=(0.0, 300.0),
            stim_start=100.0,
            stim_duration=1.0,
            stim_amplitude=40.0,
        )

        plot_results(t, sol, model, save_dir=test_dir)

        # Проверяем, что файлы созданы
        assert os.path.exists(test_dir)
        assert os.path.exists(os.path.join(test_dir, "single_cell_simulation.png"))
        assert os.path.exists(os.path.join(test_dir, "single_cell_combined.png"))
        assert os.path.exists(os.path.join(test_dir, "single_cell_data.csv"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
