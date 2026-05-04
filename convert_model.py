import tensorflow as tf
import os

# Путь к вашей модели
MODEL_PATH = r"D:\alla\image_classification_project\models\model1_tf213.h5"
OUTPUT_PATH = r"D:\alla\image_classification_project\models\animal_model.tflite"

print("🔄 Начинаем конвертацию модели...")

# Проверяем, существует ли файл
if not os.path.exists(MODEL_PATH):
    print(f"❌ Ошибка: файл {MODEL_PATH} не найден!")
    exit(1)

print(f"✅ Модель найдена. Размер: {os.path.getsize(MODEL_PATH) / (1024*1024):.1f} МБ")

# Загружаем модель
print("📦 Загрузка модели...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Создаём конвертер
print("🔄 Конвертация в TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Включаем оптимизацию "по умолчанию" (уменьшает размер)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Разрешаем стандартные и SELECT_TF_OPS (для большей совместимости)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,
]

# Конвертируем
tflite_model = converter.convert()

# Сохраняем
with open(OUTPUT_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ TFLite модель сохранена: {OUTPUT_PATH}")
print(f"📊 Размер TFLite модели: {len(tflite_model) / (1024*1024):.1f} МБ")
print(f"🎉 Готово!")