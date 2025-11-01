import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
import json

def calc_ws(gamma_wat: float) -> float:
    """
    Функция для расчета солесодержания в воде
    """
    ws = (
        1 / (gamma_wat * 1000)
        * (1.36545 * gamma_wat * 1000 - (3838.77 * gamma_wat * 1000 - 2.009 * (gamma_wat * 1000) ** 2) ** 0.5)
    )
    if ws > 0:
        return ws
    else:
        return 0

def calc_rho_w(ws: float, t: float) -> float:
    """
    Функция для расчета плотности воды в зависимости от температуры и солесодержания
    """
    rho_w = 1000 * (1.0009 - 0.7114 * ws + 0.2605 * ws ** 2) ** (-1)
    return rho_w / (1 + (t - 273) * 1e-4 * (0.269 * (t - 273) ** 0.637 - 0.8))

def calc_mu_w(ws: float, t: float, p: float) -> float:
    """
    Функция для расчета динамической вязкости воды по корреляции Matthews & Russel
    """
    a = (
        109.574
        - (0.840564 * 1000 * ws)
        + (3.13314 * 1000 * ws ** 2)
        + (8.72213 * 1000 * ws ** 3)
    )
    b = (
        1.12166
        - 2.63951 * ws
        + 6.79461 * ws ** 2
        + 54.7119 * ws ** 3
        - 155.586 * ws ** 4
    )

    mu_w = (
        a * (1.8 * t - 460) ** (-b)
        * (0.9994 + 0.0058 * (p * 1e-6) + 0.6534 * 1e-4 * (p * 1e-6) ** 2)
    )
    return mu_w

def calc_n_re(rho_w: float, q_ms: float, mu_w: float, d_tub: float) -> float:
    """
    Функция для расчета числа Рейнольдса
    """
    v = q_ms / (np.pi * d_tub ** 2 / 4)
    return rho_w * v * d_tub / mu_w * 1000

def calc_ff_churchill(n_re: float, roughness: float, d_tub: float) -> float:
    """
    Функция для расчета коэффициента трения по корреляции Churchill
    """
    if n_re == 0:
        return 0
        
    a = (-2.457 * np.log((7 / n_re) ** 0.9 + 0.27 * (roughness / d_tub))) ** 16
    b = (37530 / n_re) ** 16

    ff = 8 * ((8 / n_re) ** 12 + 1 / (a + b) ** 1.5) ** (1/12)
    return ff

def calc_dp_dl_grav(rho_w: float, angle: float) -> float:
    """
    Функция для расчета градиента на гравитацию
    """
    dp_dl_grav = rho_w * 9.81 * np.sin(angle * np.pi / 180)
    return dp_dl_grav

def calc_dp_dl_fric(rho_w: float, mu_w: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """
    Функция для расчета градиента давления на трение
    """
    if q_ms != 0:
        n_re = calc_n_re(rho_w, q_ms, mu_w, d_tub)
        ff = calc_ff_churchill(n_re, roughness, d_tub)
        dp_dl_fric = ff * rho_w * q_ms ** 2 / (d_tub ** 5 * np.pi ** 2 * 8)
    else:
        dp_dl_fric = 0
    return dp_dl_fric

def calc_dp_dl(rho_w: float, mu_w: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """
    Функция для расчета градиента давления в трубе
    """
    dp_dl_grav = calc_dp_dl_grav(rho_w, angle)
    dp_dl_fric = calc_dp_dl_fric(rho_w, mu_w, q_ms, d_tub, roughness)
    
    # Для нагнетательной скважины давление увеличивается с глубиной
    dp_dl = dp_dl_grav + dp_dl_fric
    
    return dp_dl

#####ДОБАВИЛ
def __integr_func(h: float, pt: tuple, temp_grad: float, gamma_wat: float, 
                 angle: float, q_ms: float, d_tub: float, roughness: float) -> tuple:
    """
    Функция для интегрирования по стволу скважины
    """
    p, t = pt
    
    # Расчет температуры по геотермическому градиенту
    dt_dl = temp_grad * 0.01  # преобразование в К/м
    
    # Расчет свойств воды
    ws = calc_ws(gamma_wat)
    rho_w = calc_rho_w(ws, t)
    mu_w = calc_mu_w(ws, t, p)
    
    # Расчет градиента давления
    dp_dl = calc_dp_dl(rho_w, mu_w, angle, q_ms, d_tub, roughness)
    
    return dp_dl, dt_dl

#####ДОБАВИЛ
def calc_pipe(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float, 
              gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> tuple:
    """
    Функция для расчета распределения давления и температуры по стволу скважины
    """
    # Создаем массив глубин для интегрирования
    h_points = np.linspace(h0, md_vdp, 100)
    
    # Начальные условия
    pt0 = [p_wh, t_wh]
    
    # Интегрирование системы ОДУ
    solution = solve_ivp(
        lambda h, pt: __integr_func(h, pt, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness),
        [h0, md_vdp],
        pt0,
        t_eval=h_points,
        method='RK45',
        rtol=1e-6
    )
    
    return solution.y[0], solution.y[1], solution.t

#####ДОБАВИЛ
def calc_p_wf(p_wh: float, t_wh: float, h0: float, md_vdp: float, temp_grad: float,
              gamma_wat: float, angle: float, q_ms: float, d_tub: float, roughness: float) -> float:
    """
    Функция для расчета забойного давления
    """
    p_results, t_results, h_results = calc_pipe(
        p_wh, t_wh, h0, md_vdp, temp_grad, gamma_wat, angle, q_ms, d_tub, roughness
    )
    return p_results[-1]

# Данные из варианта 12

data = {"gamma_water": 1.0436559239802052, "md_vdp": 2327.366206561479,
        "d_tub": 0.07188774060224419, "angle": 64.74462023166396,
        "roughness": 0.00034830794530518614, "p_wh": 130.13350783040906,
        "t_wh": 22.599011349724204, "temp_grad": 2.0366306785163513}

# Генерация VLP кривой
debit_range = np.linspace(1, 400, 50)  # дебиты от 1 до 400 м3/сут
p_wf_results = []

print("Расчет VLP кривой")
for i, Q in enumerate(debit_range):
    q_ms = Q / 86400  # преобразование в м3/с
    
    p_wf = calc_p_wf(
        p_wh=data["p_wh"] * 101325,  # преобразование в Па
        t_wh=data["t_wh"] + 273.15,  # преобразование в К
        h0=0,
        md_vdp=data["md_vdp"],
        temp_grad=data["temp_grad"],
        gamma_wat=data["gamma_water"],
        angle=data["angle"],
        q_ms=q_ms,
        d_tub=data["d_tub"],
        roughness=data["roughness"]
    )
    
    p_wf_atm = p_wf / 101325  # преобразование в атм
    p_wf_results.append(p_wf_atm)
    
    print(f"Дебит: {Q} м3/сут -> Забойное давление: {p_wf_atm:.2f} атм")
    
vlp_data = {
    "q_liq": debit_range.tolist(),
    "p_wf": p_wf_results
}

# Сохраняем в JSON файл
with open('vlp_results_Ovsyannikov_GM.json', 'w', encoding='utf-8') as f:
    json.dump(vlp_data, f, indent=2, ensure_ascii=False)

print(f"\nРезультаты сохранены в файл: vlp_results.json")

# Построение VLP кривой
plt.figure(figsize=(10, 6))
plt.plot(debit_range, p_wf_results, 'b-', linewidth=2, label='VLP кривая')
plt.xlabel('Дебит жидкости, м3/сут', fontsize=12)
plt.ylabel('Забойное давление, атм', fontsize=12)
plt.title('VLP нагнетательной скважины\nЗависимость забойного давления от дебита', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()

# Добавляем аннотацию с параметрами
params_text = f'Параметры скважины:\n'
params_text += f'Глубина: {data["md_vdp"]:.0f} м\n'
params_text += f'Угол: {data["angle"]:.1f}°\n'
params_text += f'Диаметр НКТ: {data["d_tub"]*1000:.1f} мм\n'
params_text += f'P_устьевое: {data["p_wh"]:.1f} атм'

plt.annotate(params_text, xy=(0.02, 0.98), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
            verticalalignment='top', fontsize=10)

plt.tight_layout()
plt.show()

# Дополнительный график - распределение давления по глубине для нескольких дебитов
plt.figure(figsize=(10, 6))
test_debits = [50, 200, 400]  # м3/сут

for Q in test_debits:
    q_ms = Q / 86400
    p_results, t_results, h_results = calc_pipe(
        p_wh=data["p_wh"] * 101325,
        t_wh=data["t_wh"] + 273.15,
        h0=0,
        md_vdp=data["md_vdp"],
        temp_grad=data["temp_grad"],
        gamma_wat=data["gamma_water"],
        angle=data["angle"],
        q_ms=q_ms,
        d_tub=data["d_tub"],
        roughness=data["roughness"]
    )
    
    plt.plot(p_results / 101325, h_results, label=f'Q = {Q} м3/сут')

plt.xlabel('Давление, атм', fontsize=12)
plt.ylabel('Глубина, м', fontsize=12)
plt.title('Распределение давления по глубине', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('pressure_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*50)
print("ОСНОВНЫЕ РЕЗУЛЬТАТЫ РАСЧЕТА VLP")
print("="*50)
print(f"Минимальное забойное давление: {min(p_wf_results):.2f} атм (при Q = {debit_range[np.argmin(p_wf_results)]:.1f} м3/сут)")
print(f"Максимальное забойное давление: {max(p_wf_results):.2f} атм (при Q = {debit_range[np.argmax(p_wf_results)]:.1f} м3/сут)")
print(f"Давление на устье: {data['p_wh']:.2f} атм")
print(f"Перепад давления (макс): {max(p_wf_results) - data['p_wh']:.2f} атм")



print("\n" + "="*50)
print("СОДЕРЖИМОЕ JSON ФАЙЛА:")
print("="*50)
print(json.dumps(vlp_data, indent=2))

# Сохраняем также текстовый файл с результатами
with open('vlp_results_Ovsyannikov_G_M.txt', 'w', encoding='utf-8') as f:
    f.write("VLP - Забойное давление и Дебит\n")
    f.write("=" * 50 + "\n")
    f.write(f"{'Q, м3/сут':<10} {'P_wf, атм':<15}\n")
    f.write("-" * 25 + "\n")
    for q, p in zip(debit_range, p_wf_results):
        f.write(f"{q:<10} {p:<15.6f}\n")
