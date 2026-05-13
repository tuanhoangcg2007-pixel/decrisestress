import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

st.set_page_config(page_title="Stress Predictor", layout="centered", page_icon="🧠")
@st.cache_resource 
def train_model():
    file_path = "stress_dataset_custom_ratio.csv"
    if not os.path.exists(file_path):
        return None, None, None

    try:
        data = pd.read_csv(file_path)
        data.columns = ['Work_Hours', 'Sleep_Hours', 'Coffee_Cups', 'Social_Media_Hours', 'Physical_Health', 'Stress_Level']
        data = data.dropna()
        x = data[['Work_Hours', 'Sleep_Hours', 'Coffee_Cups', 'Social_Media_Hours', 'Physical_Health']]
        y = data['Stress_Level']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        sc = StandardScaler()
        x_train_std = sc.fit_transform(x_train)
        model = Perceptron(max_iter=9000, eta0=0.05, random_state=42, class_weight='balanced')
        model.fit(x_train_std, y_train)
        acc = accuracy_score(y_test, model.predict(sc.transform(x_test)))
        return model, sc, acc
    except Exception as e:
        st.error(f"Lỗi: {e}")
        return None, None, None

model, sc, accuracy = train_model()
st.title("HKT APP_Ứng Dụng Giúp Bạn Dự Đoán Mức Độ Stress 🧠")

if model:
    st.sidebar.metric("Độ chính xác", f"{accuracy:.2%}")

    col1, col2 = st.columns(2)
    with col1:
        w1 = st.number_input('Số giờ làm việc:', 0.0, 24.0, 8.0, 0.5)
        w2 = st.number_input('Số giờ ngủ:', 0.0, 24.0, 7.0, 0.5)
        w3 = st.number_input('Số ly cà phê dùng trong ngày:', 0, 20, 2, 1)
    with col2:
        w4 = st.number_input('Số giờ dùng điện thoại:', 0.0, 24.0, 3.0, 0.5)
        w5 = st.slider('Tình trạng thể chất (1-10):', 1, 10, 5)

    # Kiểm tra tổng thời gian < 24h
    total_hours = w1 + w2 + w4
    if total_hours > 24:
        st.error(f"Tổng thời gian ({total_hours}h) vượt quá 24h một ngày! Vui lòng điều chỉnh lại.")
    else:
        if st.button('Dự đoán kết quả', type='primary', use_container_width=True):
            inputs_std = sc.transform([[w1, w2, w3, w4, w5]])
            res = model.predict(inputs_std)[0]

            mapping = {
                0: ("Bình thường 🟢", "#2ecc71"),
                1: ("Căng thẳng nhẹ 🟡", "#f1c40f"),
                2: ("Căng thẳng cao 🔴", "#e74c3c")
            }
            label, color = mapping.get(res, ("Không xác định", "gray"))
            
            st.divider()
            st.subheader("KẾT QUẢ DỰ ĐOÁN:")
            st.markdown(f"<h1 style='color: {color}; text-align: center;'>{label}</h1>", unsafe_allow_html=True)

            
            # 1. Cảnh báo khi dự đoán là Stress cao
            if res == 2:
                st.warning("⚠️ CẢNH BÁO: Bạn đang ở mức stress cao. Hãy nghỉ ngơi thư giãn 🧘 và điều chỉnh lịch trình ngay!")

            # 2. Cảnh báo dựa trên chỉ số thể chất (đứng độc lập)
            if w5 <= 2:
                st.warning("⚠️ CẢNH BÁO: Thể chất của bạn quá yếu khiến nguy cơ Stress cao tăng mạnh! Hãy chú ý tập thể thao 🏋️🚵🚣🏊‍♀️🧗⛹️‍♂️... NGAYYYY!!!")
else:
    st.info("Đang chờ file dữ liệu...")
