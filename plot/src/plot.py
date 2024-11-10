import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from scipy.stats import gaussian_kde
import numpy as np

# Путь к файлу с логами метрик
METRIC_LOG_PATH = r'./logs/metric_log.csv'
# Путь для сохранения графика
PLOT_OUTPUT_PATH = r'./logs/error_distribution.png'


def plot_error_distribution():
    # Создаем папку для логов, если её нет
    # os.makedirs(os.path.dirname(PLOT_OUTPUT_PATH), exist_ok=True)

    while True:
        try:
            # Загружаем данные метрик
            data = pd.read_csv(METRIC_LOG_PATH)

            if 'absolute_error' in data.columns:
                # Создаем фигуру
                plt.figure(figsize=(10, 6))

                # Строим гистограмму
                n, bins, patches = plt.hist(data['absolute_error'], bins=30,
                                            color='skyblue', edgecolor='black',
                                            alpha=0.7)

                # Добавляем линию тренда
                kde = gaussian_kde(data['absolute_error'])
                x_range = np.linspace(min(data['absolute_error']),
                                      max(data['absolute_error']), 200)
                # Масштабируем KDE к максимальной высоте гистограммы
                kde_values = kde(x_range)
                scaling_factor = max(n) / max(kde_values)
                plt.plot(x_range, kde_values * scaling_factor, 'r-', lw=2,
                         label='Trend line')

                plt.title('Distribution of Absolute Errors')
                plt.xlabel('Absolute Error')
                plt.ylabel('Count')
                plt.legend()

                # Сохраняем график в файл
                plt.savefig(PLOT_OUTPUT_PATH)
                plt.close()
            else:
                print("Column 'absolute_error' not found in metric log.")

        except Exception as e:
            print(f"Error: {e}")

        # Задержка перед следующим обновлением
        time.sleep(10)


if __name__ == "__main__":
    plot_error_distribution()
