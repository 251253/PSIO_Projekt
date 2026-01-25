from ultralytics import YOLO
import os

def main():
    # 1. Ścieżka do pliku data.yaml
    data_path = os.path.abspath("../data/yolo_training_data/data.yaml")

    print("Ładowanie modelu YOLOv8 Nano...")
    model = YOLO('yolov8n.pt')

    print(f"Rozpoczynam trening na danych z: {data_path}")
    model.train(data=data_path, epochs=50, imgsz=640, plots=True)

    print("TRENING ZAKOŃCZONY!")
    print("Model jest w folderze: runs/detect/train/weights/best.pt")

if __name__ == '__main__':
    main()