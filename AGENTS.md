# TissueModel2D

# TissueModel2D — Электромеханика сердечной ткани

## Описание проекта

Численное моделирование электромеханики сердечной ткани.
Статья: Pargaei et al. 2022, J. Mathematical Biology 84:17.

## Язык

Всегда отвечай на русском языке.

## Готовая модель клетки

- cardiac_sim/model.py — класс TNNPM (33 переменные состояния)
- cardiac_sim/calculate_parameters.py — все параметры модели
- Начальные условия: model.calculate_init_conditions()
- ВАЖНО: менять param ДО создания экземпляра TNNPM()

## Команды проекта

- Запуск тестов: pytest tests/ -v
- Запуск одной клетки: python cardiac_sim/solver.py
- Результаты сохранять в: results/

## Стек

- Python 3.10+
- scipy.integrate.solve_ivp (метод BDF)
- FEniCSx (dolfinx) для тканевой модели
- matplotlib для графиков

## Стандарты кода

- Docstrings на русском языке
- Типовые аннотации обязательны
- После каждого модуля писать тест в tests/
- Индексы вектора состояния: V=y[11], Ca_i=y[5], N=y[21]

## Параметры ишемии

- Гиперкалиемия: param["K_o"]["value"] от 5.4 до 20 мМ
- Гипоксия: param["gKATP"]["value"]

## Критерии валидации (норма)

- V_rest ≈ -86 мВ, V_peak ≈ +40 мВ
- APD90: 280–320 мс
- Пик Ca_i: 0.8–1.0 мкМ
