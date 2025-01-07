import cv2
import os

# เส้นทางไปยังโฟลเดอร์วิดีโอ
video_dirs = {
    'Anomaly': 'data/Anomaly-Videos-Part-2/',
    'Normal': 'data/Testing_Normal_Videos/'
}

# เส้นทางสำหรับเก็บเฟรม
output_dirs = {
    'Anomaly': 'data/Frames/Anomaly/',
    'Normal': 'data/Frames/Normal/'
}

# วนลูปผ่านแต่ละประเภทวิดีโอ
for category, video_dir in video_dirs.items():
    output_dir = output_dirs[category]
    os.makedirs(output_dir, exist_ok=True)
    
    for folder in os.listdir(video_dir):
        folder_path = os.path.join(video_dir, folder)
        if os.path.isdir(folder_path):  # ตรวจสอบว่าเป็นไดเรกทอรี
            for video_file in os.listdir(folder_path):
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(folder_path, video_file)
                    cap = cv2.VideoCapture(video_path)
                    count = 0
                    
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_filename = f"{os.path.splitext(video_file)[0]}_frame{count:04d}.jpg"
                        frame_path = os.path.join(output_dir, frame_filename)
                        cv2.imwrite(frame_path, frame)
                        count += 1
                    cap.release()
        elif video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, folder)
            cap = cv2.VideoCapture(video_path)
            count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_filename = f"{os.path.splitext(folder)[0]}_frame{count:04d}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                count += 1
            cap.release()

print("✅!")
