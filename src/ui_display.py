# Interfejs graficzny w Pygame

import cv2
import numpy as np

class UIDisplay:
    def __init__(self):
        self.color_accent = (245, 117, 16) # Pomarańczowy
        self.color_white = (255, 255, 255)
        self.color_bg = (30, 30, 30)

    def draw_progress_bar(self, frame, angle, min_ang=90, max_ang=160):
        """Rysuje pionowy pasek postępu na podstawie kąta rąk."""
        # Mapowanie kąta na procenty (0-100%)
        per = np.interp(angle, (min_ang, max_ang), (0, 100))
        # Mapowanie na wysokość paska (od 400 do 100 pikseli na ekranie)
        bar = np.interp(angle, (min_ang, max_ang), (400, 100))

        # Rysowanie tła paska
        cv2.rectangle(frame, (580, 100), (610, 400), self.color_bg, 3)
        # Rysowanie wypełnienia
        cv2.rectangle(frame, (580, int(bar)), (610, 400), self.color_accent, -1)
        # Napis Procentowy
        cv2.putText(frame, f'{int(per)}%', (565, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_accent, 2)
        return frame

    def draw_angle_dashboard(self, frame, angles_dict):
        """Rysuje panel z aktualnymi kątami w rogu ekranu."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (180, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        y_offset = 30
        for name, value in angles_dict.items():
            cv2.putText(frame, f"{name}: {value} deg", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_white, 1)
            y_offset += 30
        return frame