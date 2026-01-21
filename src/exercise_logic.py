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
        Analiza widoku bocznego (Kamera Telefonu).
        Sprawdza: Plecy, Nogi (Push Press) i Tor Sztangi.
        """
        self.feedback_side = ""  # Reset
        is_error = False

        # Pobranie punktów (Prawa strona ciała)
        # 12-Bark, 14-Łokieć, 16-Nadgarstek, 24-Biodro, 26-Kolano, 28-Kostka
        shoulder = [landmarks[12].x, landmarks[12].y]
        wrist = [landmarks[16].x, landmarks[16].y]
        hip = [landmarks[24].x, landmarks[24].y]
        knee = [landmarks[26].x, landmarks[26].y]
        ankle = [landmarks[28].x, landmarks[28].y]

        # 1. KĄT PLECÓW (Bark-Biodro-Kolano)
        self.back_angle = int(self.calculate_angle(shoulder, hip, knee))

        # 2. KĄT KOLANA (Biodro-Kolano-Kostka) - Detekcja Push Press
        self.knee_angle = int(self.calculate_angle(hip, knee, ankle))

        # --- LOGIKA BŁĘDÓW (Priorytety) ---

        # PRIORYTET 1: NOGI (Oszukiwanie)
        # Jeśli kąt kolana spada poniżej 165, to znaczy, że uginasz nogi
        if self.knee_angle < 165:
            self.feedback_side = "NIE UGINAJ NOG!"
            return True

        # PRIORYTET 2: PLECY (Niebezpieczeństwo)
        # Artykuł: Unikaj wypychania bioder i przeprostu lędźwiowego
        if self.back_angle < 165:
            self.feedback_side = "PROSTE PLECY!"
            return True

        # PRIORYTET 3: TOR SZTANGI (Bar Path)
        # Artykuł: Sztanga ma iść pionowo blisko ciała.
        # Sprawdzamy odległość poziomą (oś X) między barkiem a nadgarstkiem.
        # Jeśli jest duża, znaczy że sztanga "ucieka" do przodu.
        horizontal_distance = abs(shoulder[0] - wrist[0])

        # Próg 0.15 to około 15% szerokości ekranu. Dostosować w razie potrzeby.
        if horizontal_distance > 0.15 and wrist[1] < shoulder[1]:  # Tylko gdy sztanga jest nad barkami
            self.feedback_side = "SZTANGA BLIZEJ!"
            return True

        return False

    def process_front_view(self, landmarks):
        """
        Analiza widoku przedniego (Kamera Laptopa).
        """
        self.feedback_front = ""

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

        # BŁĄD: Asymetria
        wrist_diff = abs(l_wrist[1] - r_wrist[1])
        if wrist_diff > 0.08:
            self.feedback_front = "ROWNO RECE!"

        # BŁĄD: Łokcie pod sztangą
        # Jeśli łokieć jest poziomo (X) znacznie dalej niż nadgarstek
        elbow_flare_l = abs(l_elbow[0] - l_shoulder[0])
        wrist_width_l = abs(l_wrist[0] - l_shoulder[0])

        # Jeśli łokieć jest szerzej niż nadgarstek o duży margines
        if (elbow_flare_l > wrist_width_l + 0.05) and avg_angle < 120:
            self.feedback_front = "LOKCIE DO SRODKA!"

        # Liczenie
        if avg_angle > 160:
            self.current_stage = "UP"
        if avg_angle < 90 and self.current_stage == "UP":
            self.current_stage = "DOWN"
            self.reps += 1

        return self.reps, self.current_stage, self.left_arm_angle, self.right_arm_angle