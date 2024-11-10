import pika
import json
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Создаем DataFrame для хранения данных
df = pd.DataFrame(columns=['id', 'y_true', 'y_pred'])


# Имя файла для сохранения результатов
RESULTS_FILE = './logs/metric_log.csv'


def save_results():
    """Сохраняет результаты с вычисленной ошибкой в CSV файл"""
    # Вычисляем ошибку только для строк, где есть оба значения
    mask = df['y_true'].notna() & df['y_pred'].notna()
    if not mask.any():
        return

    # Создаем копию с вычисленной ошибкой
    result_df = df[mask].copy()
    result_df['absolute_error'] = abs(
        result_df['y_true'] - result_df['y_pred'])

    # Проверяем существование файла
    file_exists = os.path.isfile(RESULTS_FILE)

    # Сохраняем результаты, дополняя существующий файл
    result_df.to_csv(RESULTS_FILE,
                     mode='a',
                     header=not file_exists,
                     index=False)

    print(f"Записаны результаты в {RESULTS_FILE}:")
    print(result_df)

    # Удаляем обработанные строки из основного DataFrame
    df.drop(result_df.index, inplace=True)


def process_message(message_data, message_type):
    """Обрабатывает входящее сообщение и добавляет его в DataFrame"""
    try:
        data = json.loads(message_data)
        id_value = data.get('id')

        # Получаем значение в зависимости от типа сообщения
        if message_type == 'y_true':
            value = data.get('body')
        else:  # y_pred
            value = data.get('prediction')

        if id_value is None or value is None:
            print(f"Пропущено сообщение с некорректными данными: {data}")
            return

        # Если строка с таким id уже существует, обновляем значение
        if id_value in df.index:
            df.at[id_value, message_type] = float(value)
        else:
            # Создаем новую строку
            new_row = {'id': id_value, message_type: float(value)}
            df.loc[id_value] = pd.Series(new_row)

        print(f"Добавлено значение {message_type}: {value} для id: {id_value}")
        print(f"Текущее состояние DataFrame:\n{df}")

        # Проверяем, можно ли вычислить ошибку
        save_results()

    except json.JSONDecodeError:
        print(f"Ошибка декодирования JSON: {message_data}")
    except Exception as e:
        print(f"Ошибка обработки сообщения: {str(e)}")


def callback(ch, method, properties, body):
    """Callback-функция для обработки сообщений из очереди"""
    queue_name = method.routing_key
    message_type = 'y_true' if queue_name == 'y_true' else 'y_pred'

    print(f'Получено сообщение из очереди {queue_name}: {body.decode()}')
    answer_string = f'Из очереди {queue_name} получено значение {json.loads(body)}'
    with open('./logs/labels_log.txt', 'a') as log:
        log.write(answer_string + '\n')

    process_message(body.decode(), message_type)


def main():
    # Создаём подключение к серверу
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()

    # Объявляем очереди
    channel.queue_declare(queue='y_true')
    channel.queue_declare(queue='y_pred')

    # Настраиваем получение сообщений из обеих очередей
    channel.basic_consume(
        queue='y_true',
        on_message_callback=callback,
        auto_ack=True
    )
    channel.basic_consume(
        queue='y_pred',
        on_message_callback=callback,
        auto_ack=True
    )

    print('Ожидание сообщений... Для выхода нажмите CTRL+C')

    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        print("Получен сигнал завершения работы")
    finally:
        connection.close()


if __name__ == '__main__':
    main()
