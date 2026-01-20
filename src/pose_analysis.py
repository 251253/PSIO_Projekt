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
        Nie rysuje nic na oryginalnym obrazie (chyba że wywołasz draw=True w innej funkcji).
        """
        # MediaPipe wymaga formatu RGB, OpenCV używa BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesowanie obrazu
        results = self.pose.process(frame_rgb)

        return results

    def draw_styled_landmarks(self, frame, results):
        """
        Rysuje TYLKO kluczowe stawy (barki, łokcie, nadgarstki, biodra).
        Ignoruje twarz i palce.
        """
        if not results.pose_landmarks:
            return frame

        h, w, c = frame.shape
        landmarks = results.pose_landmarks.landmark

        # Lista połączeń, które chcemy rysować (kości)
        # Numery wg mapy MediaPipe:
        # 11-12: Barki, 11-13: Lewe ramię, 13-15: Lewe przedramię
        # 12-14: Prawe ramię, 14-16: Prawe przedramię
        # 11-23: Lewy bok, 12-24: Prawy bok, 23-24: Biodra
        connections = [
            (11, 12), (11, 13), (13, 15),
            (12, 14), (14, 16),
            (11, 23), (12, 24), (23, 24)
        ]

        # Lista punktów do narysowania (stawy)
        # 11,12=Barki, 13,14=Łokcie, 15,16=Nadgarstki, 23,24=Biodra
        indices_to_draw = [11, 12, 13, 14, 15, 16, 23, 24]

        # 1. Rysowanie Linii (Kości)
        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            # Rysujemy tylko, jeśli AI jest pewne obu punktów (visibility > 0.5)
            if start.visibility > 0.5 and end.visibility > 0.5:
                cv2.line(
                    frame,
                    (int(start.x * w), int(start.y * h)),
                    (int(end.x * w), int(end.y * h)),
                    (80, 255, 121),  # Kolor: Jasnozielony
                    3  # Grubość linii
                )

        # 2. Rysowanie Kropek (Stawy)
        for idx in indices_to_draw:
            lm = landmarks[idx]
            if lm.visibility > 0.5:
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Główna kropka
                cv2.circle(frame, (cx, cy), 6, (80, 22, 10), -1)
                # Obwódka
                cv2.circle(frame, (cx, cy), 8, (80, 255, 121), 2)

        return frame