# Создайте файл convert_model.py
import tensorflow as tf

# Загрузите вашу модель
model = tf.keras.models.load_model('models/model1_tf213.h5')

# Конвертируйте в TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Сохраните
with open('models/animal_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Модель сконвертирована! Размер:", len(tflite_model) / (1024*1024), "МБ")