from ultralytics import YOLO
import cv2
import os


class BarbellDetector:
    def __init__(self, model_path="../models/barbell.pt"):
        # Sprawdzenie czy plik istnieje, żeby uniknąć błędów
        if not os.path.exists(model_path):
            print(f"BŁĄD KRYTYCZNY: Nie znaleziono modelu pod: {model_path}")
            raise FileNotFoundError("Brak pliku modelu")

        print(f"[YOLO] Ładowanie modelu: {model_path} ...")
        self.model = YOLO(model_path)
        print("[YOLO] Model gotowy!")

    def detect(self, frame):
        """
        Zwraca środek sztangi (x, y) oraz ramkę (box).
        """
        # conf=0.4 -> Pewność minimum 40%
        results = self.model.predict(source=frame, conf=0.4, verbose=False)
        result = results[0]

        if len(result.boxes) > 0:
            # Bierzemy obiekt z największą pewnością
            box = result.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Oblicz środek
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            return (cx, cy), (x1, y1, x2, y2)

        return None, None