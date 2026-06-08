import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import sqlite3
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt  # เพิ่มบรรทัดนี้เข้ามาเพื่อใช้วาดกราฟแทนตัวเดิม

# ตั้งค่าหน้าเว็บสำหรับคนไข้
st.set_page_config(page_title="My Skin Diary", layout="centered", page_icon="✨")

# ---------------------------------------------------------
# 1. เตรียมระบบฐานข้อมูลไดอารี่ผิว (เปลี่ยนเป็น v2 เพื่อเคลียร์บั๊กไฟล์ค้าง)
# ---------------------------------------------------------
if not os.path.exists("my_skin_history"):
    os.makedirs("my_skin_history")

def init_patient_db():
    conn = sqlite3.connect("skindiary_v2.db")  # บังคับสร้างไฟล์ใหม่เบอร์ 2 ✨
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS daily_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            acne_count INTEGER,
            severity_level TEXT,
            skincare_used TEXT,
            daily_note TEXT,
            log_date DATE DEFAULT CURRENT_DATE
        )
    ''')
    conn.commit()
    conn.close()

init_patient_db()

# ---------------------------------------------------------
# 2. โหลดโมเดล AI ตรวจสิว
# ---------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("best.pt")

try:
    model = load_yolo_model()
except Exception as e:
    st.error("⚠️ ไม่พบไฟล์โมเดล 'best.pt' กรุณาตรวจสอบว่านำไฟล์มาวางในโฟลเดอร์เรียบร้อยแล้ว")
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
        return "ระดับ 1: สิวน้อยมาก (ผิวปกติ)", "🟢"
    elif count <= 15:
        return "ระดับ 2: สิวเล็กน้อย", "🟡"
    elif count <= 30:
        return "ระดับ 3: สิวปานกลาง", "🟠"
    elif count <= 50:
        return "ระดับ 4: สิวรุนแรง", "🔴"
    else:
        return "ระดับ 5: สิวรุนแรงมาก (ควรพบแพทย์)", "💀"

# ---------------------------------------------------------
# 3. เมนูสลับหน้าต่าง (Sidebar Navigation)
# ---------------------------------------------------------
st.sidebar.title("✨ My Skin Diary")
st.sidebar.write("บันทึกและติดตามสภาพผิวด้วยตัวเอง")
menu = st.sidebar.radio("เมนูใช้งาน:", ["📸 สแกนผิววันนี้", "📈 ติดตามพัฒนาการผิว"])

# =========================================================
# หน้าที่ 1: สแกนผิวและบันทึกประจำวัน
# =========================================================
if menu == "📸 สแกนผิววันนี้":
    st.title("📸 ตรวจเช็กและบันทึกสภาพผิวประจำวัน")
    st.write("ถ่ายรูปใบหน้าเพื่อประเมินความเปลี่ยนแปลงของสิวในแต่ละวัน")
    
    source_radio = st.radio("เลือกวิธีนำเข้าภาพ:", ("ถ่ายรูปด้วย Webcam", "อัปโหลดรูปภาพ"), horizontal=True)
    uploaded_image = None

    if source_radio == "อัปโหลดรูปภาพ":
        uploaded_file = st.file_uploader("เลือกรูปภาพใบหน้า...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            uploaded_image = Image.open(uploaded_file)
    else:
        cam_file = st.camera_input("กดถ่ายรูปหน้าตรง")
        if cam_file is not None:
            uploaded_image = Image.open(cam_file)

    if uploaded_image is not None:
        processed_input_image = resize_image_for_ai(uploaded_image, target_size=640)
        
        st.write("---")
        col1, col2 = st.columns(2)
        with col1:
            st.image(processed_input_image, caption="ภาพของคุณ", width="stretch")
            
        with st.spinner('AI กำลังนับเม็ดสิว...'):
            processed_img, total_acne = detect_acne_with_ai(processed_input_image)
            level_text, emoji = evaluate_severity(total_acne)
            
        with col2:
            st.image(processed_img, caption="ผลวิเคราะห์จาก AI", width="stretch")
            
        st.write("### 📊 ผลลัพธ์วันนี้")
        st.metric(label="สิวที่ตรวจพบ", value=f"{total_acne} จุด")
        st.info(f"{emoji} {level_text}")

        # โซนไดอารี่ของคนไข้
        st.write("---")
        st.write("### 📝 บันทึกไดอารี่ผิววันนี้")
        with st.form("daily_diary_form"):
            skincare = st.text_input("🧴 วันนี้ใช้สกินแคร์ / ยาแต้มสิวตัวไหนบ้าง?", placeholder="เช่น เจลแต้มสิวแบรนด์ A, มอยเจอร์ไรเซอร์")
            note = st.text_area("✍️ บันทึกเพิ่มเติม (พฤติกรรม/อาหาร/ฮอร์โมน)", placeholder="เช่น ช่วงนี้ประจำเดือนจะมา, นอนดึกตี 2, กินของทอดเยอะ")
            
            submitted = st.form_submit_button("💾 บันทึกไดอารี่ผิววันนี้")
            
            if submitted:
                today_str = datetime.date.today().strftime("%Y-%m-%d")
                saved_image_path = f"my_skin_history/skin_{today_str}.jpg"
                img_to_save = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(saved_image_path, img_to_save)
                
                # เปลี่ยนเป็น v2 ตรงกันจุดที่ 2 ✅
                conn = sqlite3.connect("skindiary_v2.db")
                c = conn.cursor()
                
                c.execute("SELECT record_id FROM daily_records WHERE log_date = CURRENT_DATE")
                existing = c.fetchone()
                
                if existing:
                    c.execute('''UPDATE daily_records 
                                 SET image_path=?, acne_count=?, severity_level=?, skincare_used=?, daily_note=? 
                                 WHERE record_id=?''', 
                              (saved_image_path, total_acne, level_text, skincare, note, existing[0]))
                else:
                    c.execute('''INSERT INTO daily_records (image_path, acne_count, severity_level, skincare_used, daily_note)
                                 VALUES (?, ?, ?, ?, ?)''', 
                              (saved_image_path, total_acne, level_text, skincare, note))
                
                conn.commit()
                conn.close()
                st.success("🎉 บันทึกไดอารี่ผิวของวันนี้เรียบร้อยแล้ว! ไปดูพัฒนาการที่หน้าเมนูด้านซ้ายได้เลย")

# =========================================================
# หน้าที่ 2: หน้าติดตามพัฒนาการ (Progress Tracking)
# =========================================================
elif menu == "📈 ติดตามพัฒนาการผิว":
    st.title("📈 พัฒนาการและแนวโน้มสภาพผิว")
    st.write("ดูสถิติการเปลี่ยนแปลงเพื่อประเมินว่าสกินแคร์ที่ใช้ได้ผลจริงหรือไม่")
    
    # เปลี่ยนเป็น v2 ตรงกันจุดที่ 3 ✅
    conn = sqlite3.connect("skindiary_v2.db")
    df = pd.read_sql_query("SELECT log_date AS 'วันที่', acne_count AS 'จำนวนสิว', severity_level AS 'ระดับ', skincare_used AS 'สกินแคร์ที่ใช้', daily_note AS 'บันทึกช่วยจำ', image_path FROM daily_records ORDER BY log_date ASC", conn)
    conn.close()
    
    if df.empty:
        st.info("ยังไม่มีข้อมูลบันทึกในระบบ เริ่มบันทึกวันแรกที่เมนู '📸 สแกนผิววันนี้' ได้เลยครับ")
    else:
        st.write("### 📉 กราฟแสดงแนวโน้มจำนวนสิว")
        
        # สร้างกราฟด้วย Matplotlib เพื่อหลบหลีกบั๊ก Python 3.14
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df['วันที่'], df['จำนวนสิว'], marker='o', color='#FF4B4B', linewidth=2, label='Acne Count')
        
        # ตกแต่งกราฟให้สวยงาม
        ax.set_title("Acne Trend Over Time", fontsize=14, pad=10)
        ax.set_xlabel("Date")
        ax.set_ylabel("Acne Count")
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # แสดงผลกราฟบน Streamlit
        st.pyplot(fig)
        
        st.write("### 📜 ประวัติบันทึกแบบละเอียด")
        st.dataframe(df.drop(columns=['image_path']), hide_index=True)
        
        st.write("---")
        st.write("### 🔍 ย้อนดูรูปภาพสกินแคร์ไดอารี่")
        selected_date = st.selectbox("เลือกวันที่ต้องการย้อนดูภาพ:", df['วันที่'].tolist())
        
        if selected_date:
            row = df[df['วันที่'] == selected_date].iloc[0]
            img_path = row['image_path']
            
            col_left, col_right = st.columns([1, 1])
            with col_left:
                if os.path.exists(img_path):
                    st.image(img_path, caption=f"สภาพผิว ณ วันที่ {selected_date}", width="stretch")
                else:
                    st.error("ไม่พบไฟล์รูปภาพ")
            with col_right:
                st.write(f"**📊 จำนวนสิวที่พบ:** {row['จำนวนสิว']} จุด")
                st.write(f"**🩺 ผลประเมิน:** {row['ระดับ']}")
                st.write(f"**🧴 สกินแคร์ที่ใช้:** {row['สกินแคร์ที่ใช้'] if row['สกินแคร์ที่ใช้'] else '-'}")
                st.write(f"**✍️ โน้ตส่วนตัว:** {row['บันทึกช่วยจำ'] if row['บันทึกช่วยจำ'] else '-'}")