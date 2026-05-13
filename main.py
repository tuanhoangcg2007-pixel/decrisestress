import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Stress Predictor", layout="centered")

# --- 1. ĐỌC DỮ LIỆU ---
@st.cache_resource # Dùng cache để không phải huấn luyện lại mỗi khi kéo thanh trượt
def train_model():
    file_path = "stress_dataset_custom_ratio.csv"
    
    if not os.path.exists(file_path):
        st.error(f"Không tìm thấy file {file_path}. Vui lòng kiểm tra lại đường dẫn!")
        return None, None, None

    data = pd.read_csv(file_path)
    data.columns = ['Work_Hours', 'Sleep_Hours', 'Coffee_Cups', 'Social_Media_Hours', 'Physical_Health', 'Stress_Level']

    # 2. Tách biến đầu vào và nhãn
    x = data[['Work_Hours', 'Sleep_Hours', 'Coffee_Cups', 'Social_Media_Hours', 'Physical_Health']]
    y = data['Stress_Level']

    # 3. Chia tập train/test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # 4. Chuẩn hóa dữ liệu
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)

    # 5. Huấn luyện Perceptron
    model = Perceptron(
        max_iter=5000,
        eta0=0.1,
        random_state=42,
        class_weight='balanced',
        tol=1e-3
    )
    model.fit(x_train_std, y_train)
    
    # Tính độ chính xác để hiển thị
    y_pred = model.predict(x_test_std)
    acc = accuracy_score(y_test, y_pred)
    
    return model, sc, acc

# Khởi tạo mô hình
model, sc, accuracy = train_model()

# --- GIAO DIỆN WEB (STREAMLIT) ---
st.title("🧠 Hệ thống dự đoán Stress (Perceptron)")
st.write("Ứng dụng sử dụng thuật toán Perceptron để phân tích mức độ căng thẳng của bạn.")

if model:
    st.sidebar.header("Thông số mô hình")
    st.sidebar.info(f"Độ chính xác: {accuracy:.2%}")

    # Layout cột để nhập liệu
    col1, col2 = st.columns(2)

    with col1:
        w1 = st.number_input('Số giờ làm việc:', value=8.0, step=0.5)
        w2 = st.number_input('Số giờ ngủ:', value=7.0, step=0.5)
        w3 = st.number_input('Số ly cà phê:', value=2.0, step=1.0)

    with col2:
        w4 = st.number_input('Số giờ dùng điện thoại:', value=3.0, step=0.5)
        w5 = st.slider('Tình trạng thể chất (1-10):', 1, 10, 5)

    # Nút dự đoán
    if st.button('Dự đoán kết quả', type='primary'):
        # Chuẩn bị dữ liệu đầu vào
        inputs = np.array([[w1, w2, w3, w4, w5]])
        inputs_std = sc.transform(inputs)

        # Dự đoán
        res = model.predict(inputs_std)[0]

        mapping = {
            0: "🟢 Bình thường (Normal)",
            1: "🟡 Căng thẳng nhẹ (Moderate)",
            2: "🔴 Căng thẳng cao (High Stress)"
        }

        # Hiển thị kết quả
        st.divider()
        st.subheader("KẾT QUẢ DỰ ĐOÁN:")
        st.header(mapping.get(res, res))

        # Logic cảnh báo sức khỏe
        if w5 <= 2:
            st.warning("⚠️ CẢNH BÁO: Thể chất yếu (<=2) khiến nguy cơ Stress cao tăng mạnh!")
        
        if res == 2:
            st.error("Bạn đang ở mức stress cao. Hãy nghỉ ngơi và thư giãn ngay!")
else:
    st.warning("Đang chờ file dữ liệu để bắt đầu...")