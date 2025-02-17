import os
import cv2
import time
import mimetypes
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO  # Utilizando YOLOv8 para detecção de objetos
from smtplib import SMTP
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

yolo_model = YOLO("modelo_treinado.pt")  

output_folder = "frames_detectados"
os.makedirs(output_folder, exist_ok=True)

last_detected_image = None
detection_log = [] 


def is_similar(image1, image2, threshold=0.90):
    if image1 is None or image2 is None:
        return False  

    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (gray2.shape[1], gray2.shape[0]))

    score, _ = ssim(gray1, gray2, full=True)
    return score >= threshold  


def detectar_objeto(image_path):
    image = Image.open(image_path)
    results = yolo_model(image)

    for result in results:
        for box in result.boxes:
            cls = result.names[int(box.cls[0])]
            if cls in ["knife", "scissors"]:  
                return True  
    return False


def enviar_alerta(image_path, tempo):
    sender = "silviosnjr@yahoo.com.br"
    receiver = "silviosnjr@gmail.com"
    subject = "Alerta de Segurança: Objeto Cortante Detectado"
    body = f"Objeto cortante detectado no tempo {tempo:.2f} segundos Veja a imagem em anexo."

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = receiver

    msg.attach(MIMEText(body, "plain"))

    try:
        with open(image_path, "rb") as img_file:
            img_data = img_file.read()
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = "application/octet-stream"

            main_type, sub_type = mime_type.split("/", 1)
            img_attachment = MIMEImage(img_data, _subtype=sub_type)
            img_attachment.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
            msg.attach(img_attachment)
    except Exception as e:
        print(f"Erro ao anexar a imagem: {e}")
        return

    smtp_server = "smtp.mail.yahoo.com"
    smtp_port = 587  
    email_password = os.getenv("SENHA_YAHOO")

    try:
        with SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # Ativa criptografia TLS
            server.login(sender, email_password)  # Login com e-mail e senha de app
            server.sendmail(sender, receiver, msg.as_string())
        print("Alerta enviado com sucesso!")
    except Exception as e:
        print("Erro ao enviar e-mail:", e)


video_path = "videos/video2.mp4" 
cap = cv2.VideoCapture(video_path)

frame_rate = cap.get(cv2.CAP_PROP_FPS) 
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    timestamp = frame_count / frame_rate
    image_path = os.path.join(output_folder, f"frame_{int(time.time())}.jpg")
    
    if last_detected_image is None or not is_similar(last_detected_image, frame, threshold=0.90):
        cv2.imwrite(image_path, frame)

        if detectar_objeto(image_path):
            detection_log.append(f"Objeto detectado no tempo: {timestamp:.2f} segundos")
            last_detected_image = frame 
            print(f"Objeto cortante detectado! Frame salvo em {image_path} - Tempo: {timestamp:.2f}s")
            enviar_alerta(image_path, timestamp)
        else:
            os.remove(image_path)
    else:
        print(f"Frame muito semelhante ao anterior, ignorado.")

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

report_path = "relatorio_deteccao.txt"
with open(report_path, "w") as report_file:
    report_file.write("Relatório de Detecção de Objetos Cortantes\n")
    report_file.write("=" * 50 + "\n")
    report_file.write("\n".join(detection_log))

print(f"Relatório salvo em {report_path}")