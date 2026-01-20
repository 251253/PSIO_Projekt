import cv2
import mediapipe as mp


class PoseAnalyzer:
    def __init__(self):
        """
        Inicjalizacja modelu MediaPipe Pose.
        """
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        # Konfiguracja modelu
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,  # False dla wideo (szybsze), True dla zdjęć
            model_complexity=1,  # 0=Lekki, 1=Średni, 2=Dokładny (ale wolny)
            smooth_landmarks=True,  # Wygładzanie drgań punktów
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def find_pose(self, frame):
        """
        Wykrywa pozę i zwraca obiekt z wynikami (landmarks).
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesowanie obrazu
        results = self.pose.process(frame_rgb)

        return results

    def draw_styled_landmarks(self, frame, results):
        """
        Rysuje szkielet: Tułów + Ręce + NOGI (Kolana i Kostki).
        Ignoruje tylko twarz i stopy (palce).
        """
        if not results.pose_landmarks:
            return frame

        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Lista połączeń (Kości)
        connections = [
            # GÓRA CIAŁA
            (11, 12), (11, 13), (13, 15),  # Barki i Lewa ręka
            (12, 14), (14, 16),  # Prawa ręka
            (11, 23), (12, 24), (23, 24),  # Tułów i Biodra

            # NOGI
            (23, 25), (25, 27),  # Lewa noga (Biodro -> Kolano -> Kostka)
            (24, 26), (26, 28)  # Prawa noga (Biodro -> Kolano -> Kostka)
        ]

        # Lista punktów (Stawy)
        indices_to_draw = [
            11, 12, 13, 14, 15, 16,  # Ręce
            23, 24,  # Biodra
            25, 26,  # Kolana (Nowe)
            27, 28  # Kostki (Nowe)
        ]

        # 1. Rysowanie Linii
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            # Rysujemy tylko, jeśli AI widzi oba punkty
            if start.visibility > 0.6 and end.visibility > 0.6:
                cv2.line(
                    frame,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (80, 255, 121),  # Jasnozielony
                    3
                )

        # 2. Rysowanie Kropek
        for idx in indices_to_draw:
            lm = landmarks[idx]
            if lm.visibility > 0.6:
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Kolana i kostki rysujemy nieco mniejsze
                radius = 6 if idx < 25 else 4

                cv2.circle(frame, (cx, cy), radius, (80, 22, 10), -1)
                cv2.circle(frame, (cx, cy), radius + 2, (80, 255, 121), 2)

        return frame