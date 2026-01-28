import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO


class BarbellDetector:
    def __init__(self, model_path):
        print("[YOLO] Ładowanie modelu w tle...")
        self.model = YOLO(model_path)

        # Zmienne do komunikacji między wątkami
        self.frame_to_process = None
        self.latest_result = (None, None)  # (center, box)
        self.stopped = False
        self.lock = threading.Lock()  # Zabezpieczenie danych
        self.new_frame_event = threading.Event()

        # Uruchamiamy wątek AI
        self.thread = threading.Thread(target=self._worker_loop)
        self.thread.daemon = True
        self.thread.start()
        print("[YOLO] Wątek gotowy.")

    def _worker_loop(self):
        """To działa w tle i mieli dane non-stop."""
        while not self.stopped:
            # Czekamy, aż główny program da nam klatkę (żeby nie palić CPU na darmo)
            self.new_frame_event.wait()

            with self.lock:
                frame = self.frame_to_process
                self.frame_to_process = None
                self.new_frame_event.clear()

            if frame is None:
                continue

            # --- ANALIZA YOLO ---
            # Tutaj dzieje się magia, która wcześniej lagowała interfejs
            try:
                results = self.model.predict(
                    source=frame,
                    conf=0.15,
                    iou=0.5,
                    agnostic_nms=True,
                    verbose=False
                )

                # Przetwarzanie wyniku
                result = results[0]
                best_center = None
                best_box = None

                if len(result.boxes) > 0:
                    max_area = 0
                    for box in result.boxes:
                        coords = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = map(int, coords)
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            best_box = (x1, y1, x2, y2)
                            best_center = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                # Zapisujemy wynik
                self.latest_result = (best_center, best_box)

            except Exception as e:
                print(f"[YOLO ERROR] {e}")

    def update_frame(self, frame):
        """Wrzuca nową klatkę do kolejki (nie blokuje programu!)."""
        with self.lock:
            self.frame_to_process = frame.copy()
            self.new_frame_event.set()  # Budzi wątek roboczy

    def get_result(self):
        """Zwraca natychmiast ostatni znany wynik."""
        return self.latest_result

    def stop(self):
        self.stopped = True
        self.new_frame_event.set()  # Wybudź, żeby mógł się zamknąć