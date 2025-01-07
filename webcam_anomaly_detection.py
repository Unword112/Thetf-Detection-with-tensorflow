import cv2
import numpy as np
from keras.models import load_model
from PIL import Image

# โหลดโมเดลที่เทรนไว้
model = load_model('model_anomaly_detection.h5')

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)  # 0 คือกล้องตัวแรก (default webcam)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # แปลงภาพเป็นขาวดำ และปรับขนาดให้ตรงกับโมเดล
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (48, 48))
    normalized_frame = resized_frame / 255.0
    image = np.expand_dims(normalized_frame, axis=0)  # เพิ่มมิติ batch
    image = np.expand_dims(image, axis=-1)  # เพิ่มมิติ channel (1 ช่องสำหรับ grayscale)

    # ทำการพยากรณ์
    prediction = model.predict(image)
    label = "Anomaly Detected" if prediction[0][0] > 0.5 else "Normal"

    if label == "Anomaly Detected":
        color = (0, 0, 255)  # สีแดงสำหรับ Anomaly
        text = "Anomaly Detected"
        # สร้างกรอบรอบพื้นที่ที่สนใจ (สมมติเป็นกรอบครอบทั้งภาพ)
        cv2.rectangle(frame, (50, 50), (frame.shape[1]-50, frame.shape[0]-50), color, 2)
    else:
        color = (0, 255, 0)  # สีเขียวสำหรับปกติ
        text = "Normal"

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Anomaly Detection', frame)

    # กด 'q' เพื่อออกจากการรัน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่างแสดงผล
cap.release()
cv2.destroyAllWindows()