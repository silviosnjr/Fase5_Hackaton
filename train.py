from ultralytics import YOLO

# 1. Carregar um modelo pré-treinado (YOLOv8s para melhor eficiência)
model = YOLO("modelo_treinado.pt")  

# 2. Treinar o modelo com seu dataset de facas
model.train(
    data="data.yaml",  # Arquivo de configuração do dataset
    epochs=50,  # Número de épocas de treinamento
    imgsz=640,  # Tamanho das imagens
    batch=16,  # Tamanho do batch
    device="cuda"  # Use "cpu" se não tiver GPU
)

# 3. Avaliar o modelo após o treinamento
metrics = model.val()

# 4. Opcional: Testar com uma imagem
result = model("train/images/10023_jpg.rf.7e6a1b5482c2eaf7fda24ef84f42fbdf.jpg")
result.show()
