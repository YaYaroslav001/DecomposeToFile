Данный скрипт строит для одной точки наблюдения графики среднегодовой температуры и количества осадков за год, диаграмму размаха среднегодовой температуры, графики декомпозиции данных по температуре и осадкам, график размаха температур, считает межквартильный размах.

Для запуска скрипта необходимы библиотеки pandas, numpy, matplotlib, seaborn и statsmodels.

Данные по температуре и осадкам берутся с помощью обновленной технологии веб-доступа АИСОРИ-М с сайта aisori-m.meteo. Датасет должен содержать все столбцы (индекс ВМО, год, месяц, день, общий признак качества температур, минимальная температура воздуха, средняя температура воздуха, максимальная температура вохдуха, количество осадков) и точку с запятой в качестве разделителя.

Перед запуском скрипта его необходимо переместить в папку с метеоданными. В командной строке необходимо перейти по пути, где лежит скрипт, с помощью команды cd. После этого запустить скрипт с одним параметром, представляющим из себя название файла с метеоданными. После завершения работы скрипта, весь результат сохраняется в папке Result.