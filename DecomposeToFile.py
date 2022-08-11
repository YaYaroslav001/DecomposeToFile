# Данный скрипт строит для одной точки наблюдения графики среднегодовой
# температуры и количества осадков за год, диаграмму размаха
# среднегодовой температуры, графики декомпозиции данных
# по температуре и осадкам, график размаха температур,
# считает межквартильный размах.

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters

# Ввод параметров.
try:
    script, data_file = argv
except ValueError:
    print("Недостаточно параметров! Необходимо параметров: 1")
    exit()

# Регистрация форматеров и конвертеров pandas в matplotlib.
register_matplotlib_converters()

# Сохранение значения стандартного вывода.
original_stdout = sys.stdout

# Считывание и обработка датасетов.
print("Считывание и обработка данных.")

# Считывание датасета.
try:
    data1 = pd.read_csv(data_file, sep=";", header=None)
except FileNotFoundError:
    print("Файл " + data1 + " не обнаружен!")
    exit()

# Присваивание имён столбцам датафрейма.
try:
    data1.columns = ["index", "year", "month", "day",
                     "temp_quality", "temp_min", "temp_avg",
                     "temp_max", "precipitation"]
except ValueError:
    print("Количество столбцов датасета не равно 9!")
    exit()

# Заполнение датафрейма.
df1 = pd.DataFrame({'year': data1["year"],
                    'month': data1["month"],
                    'day': data1["day"]})
df1["date"] = pd.to_datetime(df1)
df1["temp_avg"] = pd.to_numeric(data1["temp_avg"],
                                errors='coerce')
df1["temp_min"] = pd.to_numeric(data1["temp_min"],
                                errors='coerce')
df1["temp_max"] = pd.to_numeric(data1["temp_max"],
                                errors='coerce')
df1["precipitation"] = pd.to_numeric(data1["precipitation"],
                                     errors='coerce')

# Получение индекса метеостанции.
meteo_index = str(data1["index"].iloc[0])

# Стилизация графиков.
sns.set_style("darkgrid")
plt.rc("figure", figsize=(12, 9))
plt.rc("font", size=13)
plt.rc("lines", markersize=5)
plt.rc("lines", linewidth=3)

# Создание папки для хранения результата работы скрипта.
if not os.path.exists("Result"):
    os.makedirs("Result")

# Построения графика среднегодовой температуры.
print("Построения графика среднегодовой температуры (" + meteo_index + ").")
result1 = df1.groupby('year').mean()
plt.plot(result1.index, result1["temp_avg"])
plt.title("Среднегодовая температура (" + meteo_index + ")")
plt.xlabel('Год')
plt.ylabel('Температура (цельсии)')
z = np.polyfit(result1.index, result1['temp_avg'], 1)
p = np.poly1d(z)
plt.plot(result1.index, p(result1.index), "r--")
plt.savefig('Result/' + meteo_index + '_Temperature_Plot.png')
plt.clf()

# Построения графика количества осадков за год.
print("Построения графика количества осадков за год (" + meteo_index + ").")
plt.plot(result1.index, result1['precipitation'])
plt.title("Количество осадков за год (" + meteo_index + ")")
plt.xlabel('Год')
plt.ylabel('Количество осадков')
z = np.polyfit(result1.index, result1['precipitation'], 1)
p = np.poly1d(z)
plt.plot(result1.index, p(result1.index), "r--")
plt.savefig('Result/' + meteo_index + '_Precipitations_Plot.png')
plt.clf()

# Построения диаграммы размаха среднегодовой температуры.
print("Построения диаграммы размаха среднегодовой температуры (" + meteo_index + ").")
sns.boxplot(data=df1, x='month', y='temp_avg')
plt.xlabel('Месяц')
plt.ylabel('Температура (цельсии)')
plt.title('Температура (' + meteo_index + ')')
plt.savefig('Result/' + meteo_index + '_Temperature_Boxplot.png')
plt.clf()

# Удаление файлов с результатами декомпозиции, если они существуют.
if os.path.exists("Result/" + meteo_index + "_Temperature_Decompose.txt"):
    os.remove("Result/" + meteo_index + "_Temperature_Decompose.txt")
if os.path.exists("Result/" + meteo_index + "_Precipitations_Decompose.txt"):
    os.remove("Result/" + meteo_index + "_Precipitations_Decompose.txt")

# Декомпозиция данных по температуре.
print("Декомпозиция данных по температуре (" + meteo_index + ").")
result = seasonal_decompose(result1['temp_avg'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + meteo_index + '_Temperature_Decomposition.png')
plt.clf()
a1 = {'date': result1.index,
      'trend': result.trend,
      'seasonal': result.seasonal,
      'residual': result.resid
      }
b1 = pd.DataFrame(a1)
b1.to_csv("Result/" + meteo_index + "_Temperature_Decompose.txt", header=None,
          index=None, sep=';', mode='a')

# Декомпозиция данных по осадкам.
print("Декомпозиция данных по осадкам (" + meteo_index + ").")
result = seasonal_decompose(result1['precipitation'], model='additive', period=12)
result.plot()
plt.savefig('Result/' + meteo_index + '_Precipitations_Decomposition.png')
plt.clf()
a2 = {'date': result1.index,
      'trend': result.trend,
      'seasonal': result.seasonal,
      'residual': result.resid
      }
b2 = pd.DataFrame(a2)
b2.to_csv("Result/" + meteo_index + "_Precipitations_Decompose.txt", header=None,
          index=None, sep=';', mode='a')

# Подсчёт межквартильного размаха (IQR).
q75, q25 = np.percentile(result1['temp_avg'], [75, 25])
iqr = q75 - q25
with open('Result/' + meteo_index + '_Temperature_IQR.txt', 'w') as f:
    f.write('IQR = ' + str(iqr))
q75, q25 = np.percentile(result1['precipitation'], [75, 25])
iqr = q75 - q25
with open('Result/' + meteo_index + '_Precipitations_IQR.txt', 'w') as f:
    f.write('IQR = ' + str(iqr))

# Построение графика размаха температур.
print("Построение графика размаха температур (" + meteo_index + ").")
plt.plot(result1.index, result1['temp_max'] - result1['temp_min'])
plt.title("Размах температур (" + meteo_index + ")")
plt.xlabel('Год')
plt.ylabel('Температура (цельсии)')
z = np.polyfit(result1.index, result1['temp_max'] - result1['temp_min'], 1)
p = np.poly1d(z)
plt.plot(result1.index, p(result1.index), "r--")
plt.savefig('Result/' + meteo_index + '_TemperatureRange_Plot.png')
plt.clf()

print("\nРабота успешно завершена!")
