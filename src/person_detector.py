import cv2
from ultralytics import YOLO


class YOLOPersonDetector:
    # Klasa realizująca detekcję człowieka przy użyciu modelu YOLO

    def __init__(self, model_name="yolov8n.pt", conf=0.35, iou=0.45):
        # Inicjalizacja modelu YOLO (pobierany automatycznie przy pierwszym uruchomieniu)
        self.model = YOLO(model_name)

        # Parametry detekcji
        self.conf = conf
        self.iou = iou

        # Identyfikator klasy "person" w zbiorze COCO
        self.person_class_id = 0

    def detect_and_draw(self, frame, window_label="Human"):
        # Detekcja człowieka oraz narysowanie ramek na obrazie

        if frame is None:
            return None, 0

        # 1. Uruchomienie detekcji YOLO (tylko klasa "person")
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[self.person_class_id],
            verbose=False
        )

        # 2. Przygotowanie obrazu wyjściowego
        out = frame.copy()
        people = 0

        # 3. Przetwarzanie wykrytych obiektów
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                # Współrzędne prostokąta ograniczającego
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = float(box.conf[0])

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Rysowanie ramki
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Opis ramki (klasa + pewność)
                cv2.putText(
                    out,
                    f"{window_label}: {score:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                people += 1

        # 4. Wyświetlenie liczby wykrytych osób
        cv2.putText(
            out,
            f"People: {people}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )

        return out, people
