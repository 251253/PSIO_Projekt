import cv2
import numpy as np


class UIDisplay:
    def __init__(self):
        self.color_accent = (245, 117, 16)
        self.color_white = (255, 255, 255)
        self.color_bg = (30, 30, 30)
        self.color_btn = (50, 50, 50)
        self.color_error = (0, 0, 255)

        # Obszary przycisków dla menu
        self.btn_start_rect = [0, 0, 0, 0]
        self.btn_quit_rect = [0, 0, 0, 0]

    def combine_and_scale(self, frame_front, frame_side, target_width=1920):
        h_f, w_f = frame_front.shape[:2]
        h_s, w_s = frame_side.shape[:2]
        target_h = 720
        new_w_f = int(w_f * (target_h / h_f))
        new_w_s = int(w_s * (target_h / h_s))
        f_res = cv2.resize(frame_front, (new_w_f, target_h))
        s_res = cv2.resize(frame_side, (new_w_s, target_h))
        combined = np.hstack((f_res, s_res))
        curr_h, curr_w = combined.shape[:2]
        final_h = int(curr_h * (target_width / curr_w))
        return cv2.resize(combined, (target_width, final_h))

    def draw_advanced_menu(self, frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        cv2.putText(frame, "CyberTrener - OHP", (w // 2 - 250, h // 2 - 150),
                    cv2.FONT_HERSHEY_TRIPLEX, 2.0, self.color_accent, 3)

        bw, bh = 400, 80
        bx, by = w // 2 - bw // 2, h // 2 - 30
        self.btn_start_rect = [bx, by, bx + bw, by + bh]
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.color_btn, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), self.color_white, 2)
        cv2.putText(frame, "START [SPACJA]", (bx + 60, by + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color_white, 2)

        by_q = by + 120
        self.btn_quit_rect = [bx, by_q, bx + bw, by_q + bh]
        cv2.rectangle(frame, (bx, by_q), (bx + bw, by_q + bh), self.color_btn, -1)
        cv2.rectangle(frame, (bx, by_q), (bx + bw, by_q + bh), self.color_white, 2)
        cv2.putText(frame, "ZAKONCZ [ESC]", (bx + 70, by_q + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color_white, 2)
        return frame

    def draw_countdown(self, frame, seconds):
        h, w = frame.shape[:2]
        cv2.putText(frame, str(seconds), (w // 2 - 60, h // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 10, self.color_accent, 15)
        return frame

    def draw_workout_ui(self, frame, reps, stage, f_front, f_side, angle, angles_dict):
        h, w = frame.shape[:2]

        # 1. Górny pasek (HUD)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 130), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "POWTORZENIA", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, str(reps), (50, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.5, self.color_white, 4)
        cv2.putText(frame, "FAZA", (w // 2 - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.putText(frame, str(stage).upper(), (w // 2 - 100, 105), cv2.FONT_HERSHEY_SIMPLEX, 2, self.color_accent, 3)

        # 2. Pasek postępu (z lewej)
        bar_h = 300
        bar_y = h // 2 - 150
        per = np.interp(angle, (90, 160), (0, 100))
        fill = np.interp(angle, (90, 160), (bar_y + bar_h, bar_y))
        cv2.rectangle(frame, (40, bar_y), (70, bar_y + bar_h), (50, 50, 50), 3)
        cv2.rectangle(frame, (40, int(fill)), (70, bar_y + bar_h), self.color_accent, -1)
        cv2.putText(frame, f'{int(per)}%', (30, bar_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_accent, 2)

        # 3. Dashboard kątów (z prawej)
        dy = 180
        for name, val in angles_dict.items():
            cv2.putText(frame, f"{name}: {int(val)}", (w - 220, dy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.color_white, 2)
            dy += 40

        # 4. Panele błędów (na środku)
        self._draw_error_msg(frame, f_front, h // 2 - 60, w)
        self._draw_error_msg(frame, f_side, h // 2 + 60, w)
        return frame

    def _draw_error_msg(self, img, text, y_pos, w):
        if not text or text.strip() == "": return
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 1.2, 3)
        pad = 20
        cv2.rectangle(img, (w // 2 - tw // 2 - pad, y_pos - th // 2 - pad),
                      (w // 2 + tw // 2 + pad, y_pos + th // 2 + pad), self.color_error, -1)
        cv2.rectangle(img, (w // 2 - tw // 2 - pad, y_pos - th // 2 - pad),
                      (w // 2 + tw // 2 + pad, y_pos + th // 2 + pad), self.color_white, 2)
        cv2.putText(img, text, (w // 2 - tw // 2, y_pos + th // 2), font, 1.2, self.color_white, 3)