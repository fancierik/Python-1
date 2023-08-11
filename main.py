import numpy as np
import matplotlib.pyplot as plt
import time

# мат. функция f(x)
def f(x):
    # или можно вернуть другое значение, чтобы показать, что функция не определена для x <= 0
    return x - 5 * np.log(x)

# мат. функция фи от иск
def phi(x):
    return x - (f(x)) / (1 - 5 / x)

# производная f(x)
def derivative_f(x):
    return 1 - 5 / x

# метод отделения корней
def find_root_interval(a, b, step):
    interval_found = False
    while a <= b:
        if f(a) * f(a + step) < 0:
            interval_found = True
            break
        a += step

    if interval_found:
        return a, a + step
    else:
        print("Не удалось найти интервал с корнем.")
        return None


# метод итераций
def iteration(x0, tol, max_iter):
    start_time = time.perf_counter()
    iter_count = 0
    x = x0
    while abs(f(x)) > tol and iter_count < max_iter:
        x = phi(x)
        iter_count += 1
    execution_time = time.perf_counter() - start_time
    if abs(f(x)) <= tol:
        return x, iter_count, execution_time
    else:
        return None, iter_count, execution_time


# метод хорд
def chord_method(a, b, tol, max_iter):
    start_time = time.perf_counter()
    if f(a) * f(b) >= 0:
        print("Метод хорд не может быть применен")
        return None, 0, 0

    iter_count = 0
    x = a
    while abs(f(x)) > tol and iter_count < max_iter:
        x = x - (f(x) * (b - x)) / (f(b) - f(x))
        iter_count += 1

    execution_time = time.perf_counter() - start_time
    if abs(f(x)) <= tol:
        return x, iter_count, execution_time
    else:
        return None, iter_count, execution_time


# метод Ньютона
def newton_method(x0, tol, max_iter):
    start_time = time.perf_counter()
    iter_count = 0
    x = x0
    while abs(f(x)) > tol and iter_count < max_iter:
        x = x - f(x) / derivative_f(x)
        iter_count += 1
    execution_time = time.perf_counter() - start_time
    if abs(f(x)) <= tol:
        return x, iter_count, execution_time
    else:
        return None, iter_count, execution_time


def bisection_method(a, b, tol, max_iter):
    start_time = time.perf_counter()
    if f(a) * f(b) >= 0:
        print("Метод деления пополам не может быть применен")
        return None, 0, 0

    iter_count = 0
    while abs(b - a) > tol and iter_count < max_iter:
        c = (a + b) / 2
        if f(c) == 0:
            execution_time = time.perf_counter() - start_time
            return c, iter_count, execution_time
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        iter_count += 1

    execution_time = time.perf_counter() - start_time
    if abs(b - a) <= tol:
        return (a + b) / 2, iter_count, execution_time
    else:
        return None, iter_count, execution_time


a_scan = 0.1
b_scan = 10.0
step_size = 0.5

root_interval = find_root_interval(a_scan, b_scan, step_size)

if root_interval is not None:
    a_root, b_root = root_interval

    tolerance = 0.0001
    max_iterations = 100

    result_iteration, iterations_iteration, iteration_time_interval = iteration(a_root, tolerance, max_iterations)
    result_chord, iterations_chord, chord_time_interval = chord_method(a_root, b_root, tolerance, max_iterations)
    result_newton, iterations_newton, newton_time_interval = newton_method(a_root, tolerance, max_iterations)
    result_bisection, iterations_bisection, bisection_time_interval = bisection_method(a_root, b_root, tolerance, max_iterations)

    result_bisection_full, iterations_bisection_full, bisection_time_full = bisection_method(a_scan, b_scan, tolerance, max_iterations)
    result_iteration_full, iterations_iteration_full, iteration_time_full = iteration(a_scan, tolerance, max_iterations)
    result_chord_full, iterations_chord_full, chord_time_full = chord_method(a_scan, b_scan, tolerance, max_iterations)
    result_newton_full, iterations_newton_full, newton_time_full = newton_method(a_scan, tolerance, max_iterations)

    # print("Метод итераций:")
    # if result_iteration is not None:
    #     print("Корень: {:.6f}".format(result_iteration))
    #     # print("Количество итераций:", iterations_iteration)
    #     print("Время выполнения на интервале:", iteration_time_interval)
    # else:
    #     print("Не удалось найти корень.")
    # print()
    #
    # print("Метод хорд:")
    # if result_chord is not None:
    #     print("Корень: {:.6f}".format(result_chord))
    #     # print("Количество итераций:", iterations_chord)
    #     print("Время выполнения на интервале:", chord_time_interval)
    # else:
    #     print("Не удалось найти корень.")
    # print()
    #
    # print("Метод Ньютона:")
    # if result_newton is not None:
    #     print("Корень: {:.6f}".format(result_newton))
    #     # print("Количество итераций:", iterations_newton)
    #     print("Время выполнения на интервале:", newton_time_interval)
    # else:
    #     print("Не удалось найти корень.")
    # print()

    # создаем массив значений x и y для первого графика (от -10 до 10)
    x_full = np.linspace(-10, 10, 1000)
    y_full = f(x_full)

    # создаем массив значений x и y для второго графика (от x-0.5 до x+0.5)
    x_range = np.linspace(result_iteration - 0.5, result_iteration + 0.5, 1000)
    y_range = f(x_range)

    # строим первый график (от -10 до 10)
    plt.plot(x_full, y_full, label='f(x) = x - 5 * ln(x)')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции f(x) = x - 5 * ln(x) на интервале от 0 до 10')
    plt.legend()
    plt.grid(True)
    plt.show()

    # строим второй график (от x-0.5 до x+0.5)
    plt.figure()
    plt.plot(x_range, y_range, label='f(x) = x - 5 * ln(x)')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции f(x) в диапазоне значений аргумента х1-0.5<x<x1+0.5')
    plt.legend()
    plt.grid(True)
    plt.axhline(y=0, color='red', linestyle='--')  # Линия через ноль на оси ординат
    plt.show()

    # создаем массив значений x для временной зависимости на интервале отделения корней
    x_interval = np.arange(1, max_iterations + 1)

    # создаем массив значений x для временной зависимости на полном интервале
    x_full = np.linspace(0.01, 10, 1000)

    # создаем массив значений времени выполнения для каждого метода на интервале отделения корней
    iteration_times_interval = []
    chord_times_interval = []
    newton_times_interval = []
    bisection_times_interval = []

    # вычисляем времена выполнения для каждого количества итераций на интервале отделения корней
    for i in range(1, max_iterations + 1):
        _, _, iteration_time_interval = iteration(a_root, tolerance, i)
        _, _, chord_time_interval = chord_method(a_root, b_root, tolerance, i)
        _, _, newton_time_interval = newton_method(a_root, tolerance, i)
        _, _, bisection_time_interval = bisection_method(a_root, b_root, tolerance, i)
        iteration_times_interval.append(iteration_time_interval)
        chord_times_interval.append(chord_time_interval)
        newton_times_interval.append(newton_time_interval)
        bisection_times_interval.append(bisection_time_interval)

    # создаем массив значений времени выполнения для каждого метода на полном интервале
    iteration_times_full = []
    chord_times_full = []
    newton_times_full = []
    bisection_times_full = []

    # вычисляем времена выполнения для каждого количества итераций на полном интервале
    for i in range(1, max_iterations + 1):
        _, _, iteration_time_full = iteration(0.1, tolerance, i)
        _, _, chord_time_full = chord_method(0.1, 10, tolerance, i)
        _, _, newton_time_full = newton_method(0.1, tolerance, i)
        _, _, bisection_time_interval = bisection_method(0.1, 10, tolerance, i)
        iteration_times_full.append(iteration_time_full)
        chord_times_full.append(chord_time_full)
        newton_times_full.append(newton_time_full)
        bisection_times_full.append(bisection_time_full)

    round_decimal = 4

    # округление времени выполнения для каждого метода на интервале отделения корней
    iteration_times_interval_rounded = [round(time, round_decimal) for time in iteration_times_interval]
    chord_times_interval_rounded = [round(time, round_decimal) for time in chord_times_interval]
    newton_times_interval_rounded = [round(time, round_decimal) for time in newton_times_interval]
    bisection_times_interval_rounded = [round(time, round_decimal) for time in bisection_times_interval]

    # округление времени выполнения для каждого метода на полном интервале
    iteration_times_full_rounded = [round(time, round_decimal) for time in iteration_times_full]
    chord_times_full_rounded = [round(time, round_decimal) for time in chord_times_full]
    newton_times_full_rounded = [round(time, round_decimal) for time in newton_times_full]
    bisection_times_full_rounded = [round(time, round_decimal) for time in bisection_times_full]

    # # строим график временной зависимости для каждого метода на интервале отделения корней
    # # plt.plot(x_interval, iteration_times_interval, label='Метод итераций', color='red')
    # plt.plot(x_interval, chord_times_interval, label='Метод хорд', color='blue')
    # plt.plot(x_interval, newton_times_interval, label='Метод Ньютона', color='green')
    # plt.plot(x_interval, bisection_times_interval, label='Метод Итераций', color='red')
    # # добавляем легенду
    # plt.legend()
    #
    # # добавляем подписи к осям
    # plt.xlabel('Количество итераций')
    # plt.ylabel('Время выполнения (сек)')
    #
    # # отображаем график
    # plt.show()

    # Массив значений точностей
    tolerances = np.logspace(-7, 0, 1000)
    tolerancess = np.logspace(-4, 0, 1000)
    tolerancesss = np.logspace(-15, 0, 1000)
    # Массивы для хранения количества итераций для каждого метода на интервале отделения корней
    iterations_iteration_interval = []
    iterations_chord_interval = []
    iterations_newton_interval = []
    iterations_newton_interval1 = []
    iterations_bisection_interval = []

    # Массивы для хранения количества итераций для каждого метода на полном интервале
    iterations_iteration_full = []
    iterations_chord_full = []
    iterations_newton_full = []
    iterations_newton_full1 = []
    iterations_bisection_full = []

    for tol in tolerancess:
        _, iter_iter, _ = iteration(a_root, tol, max_iterations)
        _, iter_bisection, _ = bisection_method(a_root, b_root, tol, max_iterations)
        iterations_iteration_interval.append(iter_iter)
        iterations_bisection_interval.append(iter_bisection)
    # Вычисляем количество итераций для каждой заданной точности на интервале отделения корней
    for tol in tolerancesss:
        _, iter_chord, _ = chord_method(a_root, b_root, tol, max_iterations)
        _, iter_newton, _ = newton_method(a_root, tol, max_iterations)
        _, iter_newton1, _ = newton_method(b_root, tol, max_iterations)
        iterations_chord_interval.append(iter_chord)
        iterations_newton_interval.append(iter_newton)
        iterations_newton_interval1.append(iter_newton1)

    for tol in tolerancess:
        _, iter_iter, _ = iteration(a_scan, tol, max_iterations)
        _, iter_bisection, _ = bisection_method(a_scan, b_scan, tol, max_iterations)
        iterations_iteration_full.append(iter_iter)
        iterations_bisection_full.append(iter_bisection)


    # Вычисляем количество итераций для каждой заданной точности на полном интервале
    for tol in tolerancesss:
        _, iter_chord, _ = chord_method(a_scan, b_scan, tol, max_iterations)
        _, iter_newton, _ = newton_method(a_scan, tol, max_iterations)
        _, iter_newton1, _ = newton_method(b_root, tol, max_iterations)
        iterations_chord_full.append(iter_chord)
        iterations_newton_full.append(iter_newton)
        iterations_newton_full1.append(iter_newton1)

    # Построение графика для интервала отделения корней
    plt.figure()
    # plt.plot(tolerancess, iterations_iteration_interval, label='Метод итерации', color='red')
    plt.plot(tolerances, iterations_chord_interval, label='Метод хорд', color='red')
    plt.plot(tolerances, iterations_newton_interval, label='Метод Ньютона слева', color='green')
    plt.plot(tolerances, iterations_newton_interval1, label='Метод Ньютона справа', color='yellow')
    plt.plot(tolerances, iterations_bisection_interval, label='Метод Итераций', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Точность')
    plt.ylabel('Количество итераций')
    plt.title('График зависимости количества итераций от точности (интервал отделения корней)')
    plt.legend()
    plt.grid(True)


    # Построение графика для полного интервала
    plt.figure()
    # plt.plot(tolerancess, iterations_iteration_full, label='Метод итерации', color='red')
    plt.plot(tolerances, iterations_chord_full, label='Метод Итерации', color='blue')
    plt.plot(tolerances, iterations_newton_full, label='Метод Ньютона', color='green')
    plt.plot(tolerances, iterations_newton_interval1, label='Метод Ньютона справа', color='yellow')
    plt.plot(tolerances, iterations_bisection_full, label='Метод Хорд', color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Точность')
    plt.ylabel('Количество итераций')
    plt.title('График зависимости количества итераций от точности (полный интервал)')
    plt.legend()
    plt.grid(True)
    plt.show()
