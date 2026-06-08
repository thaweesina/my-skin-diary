import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sqlite3
import os
import datetime
import pandas as pd

# ตั้งค่าหน้าเว็บ Streamlit (ต้องอยู่บนสุด)
st.set_page_config(page_title="Clinic AI System", layout="centered", page_icon="🩺")

# ---------------------------------------------------------
# 1. ส่วนของการเตรียมฐานข้อมูลและโฟลเดอร์
# ---------------------------------------------------------
if not os.path.exists("patient_images"):
    os.makedirs("patient_images")

def init_db():
    conn = sqlite3.connect("clinic_database.db")
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS personal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            national_id TEXT,
            birthday TEXT,
            age INTEGER,
            gender TEXT,
            phone TEXT,
            emergency_phone TEXT,
            email TEXT,
            drug_allergy TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS acne_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            personal_id INTEGER,
            image_path TEXT,
            acne_count INTEGER,
            severity_level TEXT,
            check_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(personal_id) REFERENCES personal(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# ---------------------------------------------------------
# 2. ส่วนของ AI และฟังก์ชันประมวลผลภาพ
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("⚠️ ไม่พบไฟล์โมเดล 'best.pt' กรุณาตรวจสอบว่านำไฟล์จาก Colab มาวางในโฟลเดอร์เดียวกับ app2.py แล้ว")
    st.stop()

def resize_image_for_ai(image, target_size=640):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
    
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb)

def detect_acne_with_ai(image):
    results = model(image)
    res_plotted = results[0].plot() 
    img_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    acne_count = len(results[0].boxes)
    return img_rgb, acne_count

def evaluate_severity(count):
    if count <= 5:
        return "Level 1: Minimal (น้อยมาก/ผิวปกติ)", "🟢", "success"
    elif count <= 15:
        return "Level 2: Mild (เล็กน้อย)", "🟡", "info"
    elif count <= 30:
        return "Level 3: Moderate (ปานกลาง)", "🟠", "warning"
    elif count <= 50:
        return "Level 4: Severe (รุนแรง)", "🔴", "error"
    else:
        return "Level 5: Very Severe (รุนแรงมาก)", "💀", "error"

# ---------------------------------------------------------
# 3. ส่วนควบคุมเมนูนำทาง (Sidebar Navigation)
# ---------------------------------------------------------
st.sidebar.title("🏥 เมนูระบบคลินิก")
menu = st.sidebar.radio("เลือกหน้าต่างทำงาน:", ["🩺 ตรวจจับสิวและบันทึกข้อมูล", "📂 ค้นหาประวัติคนไข้"])

# =========================================================
# หน้าที่ 1: ระบบตรวจจับและบันทึกข้อมูล
# =========================================================
if menu == "🩺 ตรวจจับสิวและบันทึกข้อมูล":
    st.title("🤖 ตรวจจับและประเมินระดับสิวด้วย AI")
    st.write("อัปโหลดภาพหรือถ่ายรูป เพื่อวิเคราะห์สิวและบันทึกประวัติลงระบบ")
    
    source_radio = st.radio("เลือกช่องทางการนำเข้าภาพ:", ("Upload File ภาพ", "ใช้กล้อง Webcam"), horizontal=True)
    uploaded_image = None

    if source_radio == "Upload File ภาพ":
        uploaded_file = st.file_uploader("เลือกไฟล์ภาพใบหน้า...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
    else:
        cam_file = st.camera_input("ถ่ายภาพใบหน้าของคุณ")
        if cam_file is not None:
            uploaded_image = Image.open(cam_file)

    if uploaded_image is not None:
        processed_input_image = resize_image_for_ai(uploaded_image, target_size=640)
        
        st.write("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(processed_input_image, caption="ภาพเตรียมพร้อมสำหรับ AI", use_container_width=True)
            
        with st.spinner('AI กำลังวิเคราะห์...'):
            processed_img, total_acne = detect_acne_with_ai(processed_input_image)
            level_text, emoji, alert_type = evaluate_severity(total_acne)
            
        with col2:
            st.image(processed_img, caption="ผลการตรวจจับ (YOLOv8)", use_container_width=True)
            
        st.write("### 📊 ผลการวิเคราะห์")
        st.metric(label="จำนวนสิวที่ตรวจพบ", value=f"{total_acne} จุด")
        
        if alert_type == "success":
            st.success(f"{emoji} สรุปผลผิวหน้า: {level_text}")
        elif alert_type == "info":
            st.info(f"{emoji} สรุปผลผิวหน้า: {level_text}")
        elif alert_type == "warning":
            st.warning(f"{emoji} สรุปผลผิวหน้า: {level_text}")
        else:
            st.error(f"{emoji} สรุปผลผิวหน้า: {level_text}")

        # ฟอร์มบันทึกข้อมูลคนไข้
        st.write("---")
        st.write("### 💾 บันทึกประวัติการตรวจลงฐานข้อมูล")
        with st.form("patient_form"):
            st.write("ข้อมูลส่วนบุคคล")
            c1, c2 = st.columns(2)
            first_name = c1.text_input("ชื่อจริง *")
            last_name = c2.text_input("นามสกุล *")
            
            national_id = st.text_input("เลขประจำตัวประชาชน (13 หลัก) *")
            
            c3, c4, c5 = st.columns(3)
            birthday = c3.date_input("วันเกิด", min_value=datetime.date(1900, 1, 1))
            age = c4.number_input("อายุ", min_value=1, max_value=120)
            gender = c5.selectbox("เพศ", ["ชาย", "หญิง", "อื่นๆ"])
            
            c6, c7 = st.columns(2)
            phone = c6.text_input("เบอร์โทรศัพท์")
            emergency_phone = c7.text_input("เบอร์ติดต่อฉุกเฉิน")
            
            email = st.text_input("อีเมล")
            drug_allergy = st.text_area("ประวัติการแพ้ยา (ถ้าไม่มีให้ขีด -)")
            
            submitted = st.form_submit_button("บันทึกข้อมูลลงระบบ")
            
            if submitted:
                if not first_name or not last_name or not national_id:
                    st.warning("⚠️ กรุณากรอก ชื่อ, นามสกุล และ เลขบัตรประชาชน ให้ครบถ้วน")
                else:
                    # เซฟรูปลงโฟลเดอร์
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    saved_image_path = f"patient_images/{national_id}_{timestamp}.jpg"
                    img_to_save = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(saved_image_path, img_to_save)
                    
                    # บันทึกลงฐานข้อมูล
                    conn = sqlite3.connect("clinic_database.db")
                    c = conn.cursor()
                    
                    # ค้นหาว่ามีคนไข้คนนี้อยู่แล้วหรือไม่ (อิงจากบัตร ปชช)
                    c.execute("SELECT id FROM personal WHERE national_id = ?", (national_id,))
                    existing_patient = c.fetchone()
                    
                    if existing_patient:
                        patient_id = existing_patient[0]
                        # อัปเดตข้อมูลล่าสุด
                        c.execute('''UPDATE personal SET first_name=?, last_name=?, age=?, phone=?, email=?, drug_allergy=? WHERE id=?''', 
                                  (first_name, last_name, age, phone, email, drug_allergy, patient_id))
                    else:
                        c.execute('''INSERT INTO personal (first_name, last_name, national_id, birthday, age, gender, phone, emergency_phone, email, drug_allergy)
                                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
                                  (first_name, last_name, national_id, str(birthday), age, gender, phone, emergency_phone, email, drug_allergy))
                        patient_id = c.lastrowid
                    
                    # บันทึกประวัติสิวครั้งนี้
                    c.execute('''INSERT INTO acne_records (personal_id, image_path, acne_count, severity_level)
                                 VALUES (?, ?, ?, ?)''', 
                              (patient_id, saved_image_path, total_acne, level_text))
                    
                    conn.commit()
                    conn.close()
                    st.success("✅ บันทึกข้อมูลและรูปภาพประวัติการรักษาเรียบร้อยแล้ว!")

# =========================================================
# หน้าที่ 2: ค้นหาประวัติคนไข้
# =========================================================
elif menu == "📂 ค้นหาประวัติคนไข้":
    st.title("📂 ค้นหาประวัติคนไข้")
    st.write("ค้นหาจากชื่อ นามสกุล หรือเลขประจำตัวประชาชน เพื่อดูประวัติการตรวจทั้งหมด")
    
    search_term = st.text_input("🔍 พิมพ์คำค้นหา (เช่น ชื่อ, เลขบัตร)", "")
    
    conn = sqlite3.connect("clinic_database.db")
    
    # ดึงข้อมูลมาแสดงเป็นตาราง
    query = """
        SELECT a.record_id AS 'รหัสการตรวจ', p.first_name AS 'ชื่อ', p.last_name AS 'นามสกุล', 
               p.national_id AS 'เลขบัตร ปชช.', p.age AS 'อายุ', 
               a.check_date AS 'วัน/เวลาที่ตรวจ', a.acne_count AS 'จำนวนสิว', a.severity_level AS 'ระดับความรุนแรง', a.image_path
        FROM personal p
        JOIN acne_records a ON p.id = a.personal_id
    """
    
    if search_term:
        query += f" WHERE p.first_name LIKE '%{search_term}%' OR p.last_name LIKE '%{search_term}%' OR p.national_id LIKE '%{search_term}%'"
        
    query += " ORDER BY a.check_date DESC"
    
    df = pd.read_sql_query(query, conn)
    
    if df.empty:
        st.info("ไม่พบข้อมูลในระบบ หรือยังไม่มีการบันทึกประวัติ")
    else:
        # แสดงตาราง (ซ่อนคอลัมน์ image_path ไว้ไม่ให้รกตา)
        st.dataframe(df.drop(columns=['image_path']), use_container_width=True)
        
        st.write("---")
        st.write("### 🖼️ ดูรูปภาพประวัติการตรวจ")
        selected_record = st.selectbox("เลือกรหัสการตรวจที่ต้องการดูภาพ:", df['รหัสการตรวจ'].tolist())
        
        if selected_record:
            # ดึง path ภาพของ record นั้นมาแสดง
            img_path = df[df['รหัสการตรวจ'] == selected_record]['image_path'].values[0]
            level_show = df[df['รหัสการตรวจ'] == selected_record]['ระดับความรุนแรง'].values[0]
            count_show = df[df['รหัสการตรวจ'] == selected_record]['จำนวนสิว'].values[0]
            
            if os.path.exists(img_path):
                st.image(img_path, caption=f"ภาพผลการตรวจ - พบสิว {count_show} จุด ({level_show})", width=400)
            else:
                st.error("⚠️ ไม่พบไฟล์รูปภาพในโฟลเดอร์ อาจถูกลบหรือเคลื่อนย้าย")

    conn.close()