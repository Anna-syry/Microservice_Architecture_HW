import pika
import numpy as np
import json
from sklearn.datasets import load_diabetes
import time
from datetime import datetime

np.random.seed(42)
# Загружаем датасет о диабете
X, y = load_diabetes(return_X_y=True)


# Подключение к серверу на локальном хосте:
connection = pika.BlockingConnection(pika.ConnectionParameters('rabbitmq'))
channel = connection.channel()

# Создаём очередь y_true
channel.queue_declare(queue='y_true')
# Создаём очередь features
channel.queue_declare(queue='features')


while (True):

    # Формируем случайный индекс строки
    random_row = np.random.randint(0, X.shape[0]-1)
    message_id = datetime.timestamp(datetime.now())

    # Формируем сообщение с идентификатором
    message_y_true = {
        'id': message_id,
        'body': y[random_row]
    }

    # Публикуем сообщение в очередь y_true
    channel.basic_publish(exchange='',
                          routing_key='y_true',
                          body=json.dumps(message_y_true))
    print(
        f'Сообщение id {message_id} с правильным ответом отправлено в очередь')

    # Формируем сообщение с идентификатором
    message_x = {
        'id': message_id,
        'body': list(X[random_row])
    }

    # Публикуем сообщение в очередь features
    channel.basic_publish(exchange='',
                          routing_key='features',
                          body=json.dumps(message_x))
    print(
        f'Сообщение id {message_id} с вектором признаков отправлено в очередь')
    time.sleep(5)
# Закрываем подключение
connection.close()
