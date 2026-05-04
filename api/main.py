import os
import io
import random
import numpy as np
from PIL import Image
import tensorflow as tf  # Это нужно только для tf.lite, он есть в tflite-runtime
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

# 🔧 Пути к моделям
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

ANIMAL_MODEL_PATH = os.path.join(MODELS_DIR, "animal_model.tflite")
MNIST_MODEL_PATH = os.path.join(MODELS_DIR, "mnist_model.tflite")

ANIMAL_CLASSES = ['Кошка', 'Собака', 'Панда']

# ================= ЗАГРУЗКА МОДЕЛЕЙ =================
print("📦 Загрузка моделей...")

# TFLite модель для животных
animal_interpreter = None
animal_loaded = False
if os.path.exists(ANIMAL_MODEL_PATH):
    try:
        animal_interpreter = tf.lite.Interpreter(model_path=ANIMAL_MODEL_PATH)
        animal_interpreter.allocate_tensors()
        print(f"✅ TFLite модель животных загружена: {os.path.basename(ANIMAL_MODEL_PATH)}")
        animal_loaded = True
    except Exception as e:
        print(f"❌ Ошибка загрузки TFLite модели животных: {e}")
else:
    print(f"⚠️ Файл модели животных не найден: {ANIMAL_MODEL_PATH}")

# MNIST модель (TFLite)
mnist_interpreter = None
mnist_loaded = False
if os.path.exists(MNIST_MODEL_PATH):
    try:
        mnist_interpreter = tf.lite.Interpreter(model_path=MNIST_MODEL_PATH)
        mnist_interpreter.allocate_tensors()
        print(f"✅ MNIST модель загружена: {os.path.basename(MNIST_MODEL_PATH)}")
        mnist_loaded = True
    except Exception as e:
        print(f"❌ Ошибка загрузки MNIST: {e}")
else:
    print(f"⚠️ Файл MNIST модели не найден: {MNIST_MODEL_PATH}")

# ================= ПРЕПРОЦЕССИНГ =================
def preprocess_animal_image(image, target_size=(128, 128)):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def preprocess_mnist_image(image):
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(image, dtype=np.float32)
    # Инверсия, если фон светлый
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    img_array = img_array / 255.0
    return img_array.reshape(1, 28, 28, 1).astype(np.float32)

# ================= ЭНДПОИНТЫ =================
@app.get("/")
async def root():
    return {
        "status": "running",
        "animal_model": animal_loaded,
        "mnist_model": mnist_loaded
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict/animal/{model_name}")
async def predict_animal(model_name: str, file: UploadFile = File(...)):
    # Если модель не загружена — демо-режим
    if not animal_loaded or animal_interpreter is None:
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
        
        # TFLite инференс
        input_details = animal_interpreter.get_input_details()
        output_details = animal_interpreter.get_output_details()
        animal_interpreter.set_tensor(input_details[0]['index'], processed)
        animal_interpreter.invoke()
        predictions = animal_interpreter.get_tensor(output_details[0]['index'])[0]
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
    if not mnist_loaded or mnist_interpreter is None:
        return JSONResponse(status_code=500, content={"error": "MNIST model not loaded"})
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        processed = preprocess_mnist_image(image)
        
        input_details = mnist_interpreter.get_input_details()
        output_details = mnist_interpreter.get_output_details()
        mnist_interpreter.set_tensor(input_details[0]['index'], processed)
        mnist_interpreter.invoke()
        predictions = mnist_interpreter.get_tensor(output_details[0]['index'])[0]
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