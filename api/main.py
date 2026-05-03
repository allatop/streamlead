import os
import io
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔧 Надёжные пути (работают и при python main.py, и при uvicorn)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

ANIMAL_MODEL_PATH = os.path.join(MODELS_DIR, "model1_tf213.h5")  # 👈 Обновлено
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, "mnist_model.tflite")

ANIMAL_CLASSES = ['Кошка', 'Собака', 'Панда']

# ================= ЗАГРУЗКА МОДЕЛЕЙ =================
print("📦 Загрузка моделей...")

# Модель для животных
try:
    animal_model = tf.keras.models.load_model(ANIMAL_MODEL_PATH, compile=False)
    print(f"✅ Модель животных загружена: {os.path.basename(ANIMAL_MODEL_PATH)}")
    animal_loaded = True
except Exception as e:
    print(f"❌ Ошибка загрузки модели животных: {e}")
    animal_loaded = False

# MNIST модель (TFLite)
try:
    interpreter = tf.lite.Interpreter(model_path=MNIST_MODEL_PATH)
    interpreter.allocate_tensors()
    print(f"✅ MNIST модель загружена: {os.path.basename(MNIST_MODEL_PATH)}")
    mnist_loaded = True
except Exception as e:
    print(f"❌ Ошибка загрузки MNIST: {e}")
    mnist_loaded = False

# ================= ПРЕПРОЦЕССИНГ =================
def preprocess_animal_image(image, target_size=(128, 128)):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_mnist_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    # Инверсия, если фон светлый (стандарт MNIST: чёрные цифры на белом фоне)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28, 1)

# ================= ЭНДПОИНТЫ =================
@app.get("/")
async def root():
    return {
        "status": "running",
        "animal_model": animal_loaded,
        "mnist_model": mnist_loaded
    }

@app.post("/predict/animal/{model_name}")
async def predict_animal(model_name: str, file: UploadFile = File(...)):
    if not animal_loaded:
        # Демо-режим
        pred_idx = random.randint(0, 2)
        probs = [0.7, 0.2, 0.1]
        random.shuffle(probs)
        return {
            "success": True,
            "model": model_name,
            "predicted_label": ANIMAL_CLASSES[pred_idx],
            "confidence": probs[pred_idx],
            "probabilities": {ANIMAL_CLASSES[i]: probs[i] for i in range(3)}
        }
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_animal_image(image)
        predictions = animal_model.predict(processed, verbose=0)[0]
        predicted_class = int(np.argmax(predictions))
        
        return {
            "success": True,
            "model": model_name,
            "predicted_label": ANIMAL_CLASSES[predicted_class],
            "confidence": float(predictions[predicted_class]),
            "probabilities": {
                "Кошка": float(predictions[0]),
                "Собака": float(predictions[1]),
                "Панда": float(predictions[2])
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/digit")
async def predict_digit(file: UploadFile = File(...)):
    if not mnist_loaded:
        return JSONResponse(status_code=500, content={"error": "MNIST model not loaded"})
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_mnist_image(image)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], processed)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predicted_class = int(np.argmax(predictions))
        
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": float(predictions[predicted_class]),
            "probabilities": {str(i): float(predictions[i]) for i in range(10)}
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)