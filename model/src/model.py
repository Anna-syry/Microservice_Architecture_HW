import pika
import pickle
import numpy as np
import json
from datetime import datetime

# Читаем файл с сериализованной моделью
with open('myfile.pkl', 'rb') as pkl_file:
    regressor = pickle.load(pkl_file)

# Создаём подключение к серверу на локальном хосте
connection = pika.BlockingConnection(
    pika.ConnectionParameters(host='rabbitmq'))
channel = connection.channel()

# Объявляем очереди
channel.queue_declare(queue='features')
channel.queue_declare(queue='y_pred')  # Очередь для отправки предсказаний

# Функция callback для обработки данных из очереди features


def callback(ch, method, properties, body):
    # Десериализация сообщения
    message = json.loads(body)

    # Извлечение уникального идентификатора из сообщения
    message_id = message['id']

    # Извлечение признаков из сообщения
    features = message['body']

    # Преобразование признаков в numpy массив нужной формы
    shaped_features = np.array(features).reshape(1, -1)

    # Предсказание модели
    pred = regressor.predict(shaped_features)
    prediction = round(pred[0])  # Округление до целого

    # Формирование сообщения с предсказанием и идентификатором
    prediction_message = {
        'id': message_id,  # Используем идентификатор из входящего сообщения
        'prediction': prediction
    }

    # Сериализация и отправка предсказания в очередь y_pred
    channel.basic_publish(
        exchange='', routing_key='y_pred', body=json.dumps(prediction_message))

    print(
        f'Предсказание {prediction} с id {message_id} отправлено в очередь y_pred')


# Настройка потребителя для очереди features
channel.basic_consume(
    queue='features',
    on_message_callback=callback,
    auto_ack=True
)

print('...Ожидание сообщений, для выхода нажмите CTRL+C')
channel.start_consuming()
