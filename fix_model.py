import h5py
import json
import sys

MODEL_PATH = "D:/alla/image_classification_project/models/model1_tf213.h5"

print("🔧 Патчим модель для Keras 2.13...")

# Ключи, которые есть в Keras 3.x, но отсутствуют в Keras 2.13
KERAS3_KEYS_TO_REMOVE = [
    'optional',           # InputLayer
    'registered_name',    # все слои
    'quantization_config', # Dense, Conv2D и др.
]

def convert_keras3_to_keras2(obj):
    """Рекурсивно конвертирует конфиг модели: Keras 3.x → Keras 2.13"""
    if isinstance(obj, dict):
        # 1. DTypePolicy → строка (например, "float32")
        if 'dtype' in obj and isinstance(obj['dtype'], dict):
            dtype_info = obj['dtype']
            if dtype_info.get('class_name') == 'DTypePolicy':
                obj['dtype'] = dtype_info.get('config', {}).get('name', 'float32')
        
        # 2. InputLayer: batch_shape → batch_input_shape
        if 'batch_shape' in obj:
            obj['batch_input_shape'] = obj.pop('batch_shape')
        
        # 3. Удаляем ключи, неизвестные Keras 2.13
        for key in KERAS3_KEYS_TO_REMOVE:
            obj.pop(key, None)
        
        # 4. Рекурсивно обрабатываем вложенные объекты
        for key, value in list(obj.items()):
            if isinstance(value, (dict, list)):
                convert_keras3_to_keras2(value)
                
    elif isinstance(obj, list):
        for item in obj:
            convert_keras3_to_keras2(item)

with h5py.File(MODEL_PATH, 'r+') as f:
    if 'model_config' not in f.attrs:
        print("❌ Ошибка: файл не содержит конфиг модели Keras.")
        sys.exit(1)
        
    config_str = f.attrs['model_config']
    if isinstance(config_str, bytes):
        config_str = config_str.decode('utf-8')
        
    config = json.loads(config_str)
    convert_keras3_to_keras2(config)
    
    # Записываем исправленный конфиг обратно в файл
    f.attrs.modify('model_config', json.dumps(config).encode('utf-8'))
    print("✅ Модель успешно сконвертирована для Keras 2.13!")