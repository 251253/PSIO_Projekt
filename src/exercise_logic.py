import numpy as np


class ExerciseLogic:
    def __init__(self):
        # Stan licznika
        self.reps = 0
        self.current_stage = "DOWN"

        # Zmienne debugowania (kąty)
        self.left_arm_angle = 0
        self.right_arm_angle = 0
        self.back_angle = 0
        self.knee_angle = 0

        # Komunikaty
        self.feedback_front = ""
        self.feedback_side = ""

    def calculate_angle(self, a, b, c):
        """Oblicza kąt w wierzchołku b."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle
        return angle

    def check_side_errors(self, landmarks):
        """
        Analiza widoku bocznego (Kamera Telefonu) - Tylko MediaPipe.
        """
        self.feedback_side = ""

        # 12-Bark, 24-Biodro, 26-Kolano, 28-Kostka
        shoulder = [landmarks[12].x, landmarks[12].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        hip = [landmarks[24].x, landmarks[24].y]
        knee = [landmarks[26].x, landmarks[26].y]
        ankle = [landmarks[28].x, landmarks[28].y]

        # 1. KĄT PLECÓW
        self.back_angle = int(self.calculate_angle(shoulder, hip, knee))

        # 2. KĄT KOLANA
        self.knee_angle = int(self.calculate_angle(hip, knee, ankle))

        # --- LOGIKA BŁĘDÓW ---
        if self.knee_angle < 155:
            self.feedback_side = "NIE UGINAJ NOG!"
            return True

        if self.back_angle < 160:
            self.feedback_side = "PROSTE PLECY!"
            return True

        # Sprawdzenie czy sztanga nie jest za daleko (bazując na nadgarstkach)
        horizontal_distance = abs(shoulder[0] - wrist[0])
        if horizontal_distance > 0.15 and wrist[1] < shoulder[1]:
            self.feedback_side = "SZTANGA BLIZEJ!"
            return True

        return False

    def process_front_view(self, landmarks, bar_center_x=None, frame_width=1280):
        """
        Analiza widoku przedniego (Kamera Laptopa).
        Teraz przyjmuje też pozycję sztangi (bar_center_x) z YOLO!
        """
        self.feedback_front = ""

        # Pobieramy punkty
        nose = [landmarks[0].x, landmarks[0].y]  # Nos jest pod indeksem 0

        l_shoulder = [landmarks[11].x, landmarks[11].y]
        l_elbow = [landmarks[13].x, landmarks[13].y]
        l_wrist = [landmarks[15].x, landmarks[15].y]
        r_shoulder = [landmarks[12].x, landmarks[12].y]
        r_elbow = [landmarks[14].x, landmarks[14].y]
        r_wrist = [landmarks[16].x, landmarks[16].y]

        angle_l = self.calculate_angle(l_shoulder, l_elbow, l_wrist)
        angle_r = self.calculate_angle(r_shoulder, r_elbow, r_wrist)

        self.left_arm_angle = int(angle_l)
        self.right_arm_angle = int(angle_r)
        avg_angle = (angle_l + angle_r) / 2

        # --- NOWOŚĆ: SPRAWDZANIE ŚRODKA CIĘŻKOŚCI (YOLO vs NOS) ---
        if bar_center_x is not None:
            # Przeliczamy pozycję nosa na piksele
            nose_pixel_x = nose[0] * frame_width

            # Obliczamy różnicę
            diff = abs(nose_pixel_x - bar_center_x)

            # Jeśli różnica większa niż 8% szerokości ekranu -> BŁĄD
            if diff > (frame_width * 0.08):
                self.feedback_front = "SRODEK!"

        # Standardowe błędy MediaPipe (nadal działają)
        wrist_diff = abs(l_wrist[1] - r_wrist[1])
        if wrist_diff > 0.08 and self.feedback_front == "":
            self.feedback_front = "ROWNO RECE!"

        elbow_flare_l = abs(l_elbow[0] - l_shoulder[0])
        wrist_width_l = abs(l_wrist[0] - l_shoulder[0])
        if (elbow_flare_l > wrist_width_l + 0.05) and avg_angle < 120 and self.feedback_front == "":
            self.feedback_front = "LOKCIE DO SRODKA!"

        # Liczenie powtórzeń
        if avg_angle > 160:
            self.current_stage = "UP"
        if avg_angle < 90 and self.current_stage == "UP":
            self.current_stage = "DOWN"
            self.reps += 1

        return self.reps, self.current_stage, self.left_arm_angle, self.right_arm_angle