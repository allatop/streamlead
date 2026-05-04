import streamlit as st
import requests
from PIL import Image
import io
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="Классификатор изображений", layout="wide")

API_URL = "https://streamlead.onrender.com"

st.title("🐾 Классификатор изображений")
st.markdown("---")

# Результаты сравнения моделей
st.header("📊 Результаты сравнения моделей")

results_data = {
    "Модель": ["Model 1", "Model 3", "Model 5"],
    "Точность": ["71.68%", "70.80%", "32.74%"],
    "F1-мера": ["70.53%", "70.46%", "16.15%"],
    "Статус": ["✅ Лучшая", "👍 Хорошая", "❌ Слабая"]
}
df = pd.DataFrame(results_data)
st.dataframe(df, hide_index=True)

# График сравнения
fig, ax = plt.subplots(figsize=(10, 5))
models = ["Model 1", "Model 3", "Model 5"]
acc = [71.68, 70.80, 32.74]
f1 = [70.53, 70.46, 16.15]

x = range(len(models))
width = 0.35
bars1 = ax.bar([i - width/2 for i in x], acc, width, label='Точность', color='#4CAF50')
bars2 = ax.bar([i + width/2 for i in x], f1, width, label='F1-мера', color='#2196F3')

ax.set_xlabel('Модели')
ax.set_ylabel('Проценты (%)')
ax.set_title('Сравнение качества моделей')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.set_ylim(0, 100)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center', fontsize=10)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}%', ha='center', fontsize=10)

st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# Выбор задачи
task = st.radio("Выберите задачу:", ["🐕 Распознавание животных", "🔢 Распознавание цифр"], horizontal=True)
st.markdown("---")

# ============================================
# РАСПОЗНАВАНИЕ ЖИВОТНЫХ
# ============================================
if task == "🐕 Распознавание животных":
    st.subheader("🐕 Распознавание животных: Кошка, Собака, Панда")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_choice = st.selectbox("Выберите модель:", ["Model 1 (Лучшая)", "Model 3", "Model 5"])
        if "1" in model_choice:
            model_name = "model1"
        elif "3" in model_choice:
            model_name = "model3"
        else:
            model_name = "model5"
        
        input_method = st.radio("Способ ввода:", ["📁 Загрузить фото", "📷 Сделать фото"], horizontal=True)
        
        image = None
        
        if input_method == "📁 Загрузить фото":
            uploaded = st.file_uploader("Выберите изображение", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                st.image(image, caption="Ваше изображение", use_column_width=True)
        else:
            camera = st.camera_input("Сделайте фото")
            if camera:
                image = Image.open(camera)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                st.image(image, caption="Ваше фото", use_column_width=True)
    
    with col2:
        if image and st.button("🔍 Распознать", type="primary"):
            with st.spinner("Анализируем изображение..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='JPEG', quality=95)
                img_bytes = img_bytes.getvalue()
                
                try:
                    response = requests.post(
                        f"{API_URL}/predict/animal/{model_name}",
                        files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        r = response.json()
                        if r.get('success'):
                            label = r['predicted_label']
                            confidence = r['confidence']
                            
                            colors = {"Кошка": "#FF6B6B", "Собака": "#4ECDC4", "Панда": "#45B7D1"}
                            
                            st.markdown(f"""
                            <div style='text-align: center; padding: 20px; background: {colors[label]}20; border-radius: 20px; border: 2px solid {colors[label]};'>
                                <h2>Результат:</h2>
                                <h1 style='font-size: 48px;'>{label}</h1>
                                <p style='font-size: 18px;'>Уверенность: {confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.subheader("📊 Вероятности по классам")
                            probs = r['probabilities']
                            fig, ax = plt.subplots(figsize=(6, 3))
                            classes = list(probs.keys())
                            values = [probs[c] * 100 for c in classes]
                            colors_list = [colors[c] for c in classes]
                            ax.barh(classes, values, color=colors_list)
                            ax.set_xlabel('Вероятность (%)')
                            ax.set_xlim(0, 100)
                            for i, v in enumerate(values):
                                ax.text(v + 1, i, f'{v:.1f}%', va='center')
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.error("Ошибка при распознавании")
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка подключения: {e}")

# ============================================
# РАСПОЗНАВАНИЕ ЦИФР
# ============================================
else:
    st.subheader("🔢 Распознавание рукописных цифр (0-9)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_method = st.radio(
            "Способ ввода:",
            ["✏️ Нарисовать цифру", "📁 Загрузить фото", "📷 Сделать фото"],
            horizontal=True
        )
        
        image = None
        
        if input_method == "✏️ Нарисовать цифру":
            st.markdown("**Нарисуйте цифру на холсте**")
            stroke_width = st.slider("Толщина линии:", 10, 50, 25)
            
            col_color1, col_color2 = st.columns(2)
            with col_color1:
                stroke_color = st.color_picker("Цвет линии:", "#FFFFFF")
            with col_color2:
                bg_color = st.color_picker("Цвет фона:", "#000000")
            
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_color=bg_color,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="digit_canvas",
            )
            if canvas_result.image_data is not None:
                image = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
                image = image.convert('L')
                st.image(image, caption="Ваш рисунок", use_column_width=True)
        
        elif input_method == "📁 Загрузить фото":
            uploaded = st.file_uploader("Выберите изображение с цифрой", type=["jpg", "jpeg", "png"])
            if uploaded:
                image = Image.open(uploaded)
                if image.mode != 'L':
                    image = image.convert('L')
                st.image(image, caption="Ваше изображение", use_column_width=True)
        else:
            camera = st.camera_input("Сфотографируйте цифру")
            if camera:
                image = Image.open(camera)
                if image.mode != 'L':
                    image = image.convert('L')
                st.image(image, caption="Ваше фото", use_column_width=True)
    
    with col2:
        if image and st.button("🔢 Распознать цифру", type="primary"):
            with st.spinner("Анализируем..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                try:
                    response = requests.post(
                        f"{API_URL}/predict/digit",
                        files={"file": ("digit.png", img_bytes, "image/png")},
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        r = response.json()
                        pred = r['predicted_class']
                        confidence = r['confidence']
                        
                        st.markdown(f"""
                        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px;'>
                            <h2>Результат:</h2>
                            <h1 style='font-size: 80px; color: white;'>{pred}</h1>
                            <p style='font-size: 18px; color: white;'>Уверенность: {confidence:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.subheader("📊 Вероятности по цифрам")
                        probs = r['probabilities']
                        fig, ax = plt.subplots(figsize=(10, 4))
                        digits = list(probs.keys())
                        vals = [probs[d] * 100 for d in digits]
                        colors_d = ['#4CAF50' if int(d) == pred else '#E0E0E0' for d in digits]
                        bars = ax.bar(digits, vals, color=colors_d)
                        ax.set_xlabel('Цифра')
                        ax.set_ylabel('Вероятность (%)')
                        ax.set_ylim(0, 100)
                        for bar, v in zip(bars, vals):
                            ax.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%', ha='center', fontsize=9)
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.error(f"Ошибка API: {response.status_code}")
                except Exception as e:
                    st.error(f"Ошибка подключения: {e}")

st.markdown("---")
