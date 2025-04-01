import cv2
import mediapipe as mp
import numpy as np
import time

class HandsFreeController:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.frame_count = 0
        self.last_time = time.time()
        self.gaze_history = []  # For smoothing

    def draw_holographic_ui(self, frame, gaze_x, gaze_y):
        h, w, _ = frame.shape
        overlay = np.zeros_like(frame, dtype=np.uint8)  # Transparent overlay

        # Holographic background (scanlines + grid)
        for y in range(0, h, 20):
            cv2.line(overlay, (0, y), (w, y), (0, 50, 100), 1)  # Faint cyan lines
        for x in range(0, w, 20):
            cv2.line(overlay, (x, 0), (x, h), (0, 50, 100), 1)
        alpha = 0.2 + 0.1 * np.sin(self.frame_count * 0.05)  # Pulsing opacity
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Dynamic gaze zones with particle effects
        zones = {
            "top": ((w//4, 0), (3*w//4, h//4), (0, 255, 200)),      # Cyan
            "bottom": ((w//4, 3*h//4), (3*w//4, h), (255, 0, 100)), # Magenta
            "left": ((0, h//4), (w//4, 3*h//4), (100, 0, 255)),     # Purple
            "right": ((3*w//4, h//4), (w, 3*h//4), (255, 200, 0))   # Orange
        }

        gaze_px, gaze_py = int(gaze_x * w), int(gaze_y * h)
        for name, ((x1, y1), (x2, y2), color) in zones.items():
            # Animated border
            intensity = 50 + int(50 * np.sin(self.frame_count * 0.1))
            border_color = tuple(min(c + intensity, 255) for c in color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

            # Highlight active zone with particles
            if (x1 <= gaze_px <= x2) and (y1 <= gaze_py <= y2):
                cv2.putText(frame, f"[{name.upper()}]", (x1 + 10, y1 + 30), 
                            cv2.FONT_HERSHEY_TRIPLEX, 0.8, color, 2)
                # Particle effect: random glowing dots
                for _ in range(5):
                    px = np.random.randint(x1, x2)
                    py = np.random.randint(y1, y2)
                    cv2.circle(frame, (px, py), 2, color, -1)

        # Advanced gaze reticle (3D-like)
        cv2.circle(frame, (gaze_px, gaze_py), 15, (0, 255, 255), 1)  # Outer orbit
        cv2.circle(frame, (gaze_px, gaze_py), 10, (0, 200, 255), 1)  # Inner orbit
        # Rotating arcs
        angle = self.frame_count * 5 % 360
        for r in [20, 25]:
            cv2.ellipse(frame, (gaze_px, gaze_py), (r, r), 0, angle, angle + 90, (0, 255, 200), 1)
            cv2.ellipse(frame, (gaze_px, gaze_py), (r, r), 0, angle + 180, angle + 270, (0, 255, 200), 1)

        # HUD: Floating status panel
        panel_y = h - 60
        cv2.rectangle(frame, (10, panel_y), (w - 10, h - 10), (20, 20, 50), -1)  # Dark base
        cv2.rectangle(frame, (10, panel_y), (w - 10, h - 10), (0, 255, 200), 1)  # Cyan border
        fps = 1 / (time.time() - self.last_time)
        self.last_time = time.time()
        cv2.putText(frame, f"GAZE: X={gaze_x:.2f}, Y={gaze_y:.2f} | FPS: {fps:.1f}", 
                    (20, h - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 200), 1)
        cv2.putText(frame, "NEURAL INTERFACE ACTIVE", (w - 220, h - 30), 
                    cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 200), 1)

        return frame

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        self.frame_count += 1

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                gaze_x = (left_eye.x + right_eye.x) / 2
                gaze_y = (left_eye.y + right_eye.y) / 2
                # Smooth gaze with history
                self.gaze_history.append((gaze_x, gaze_y))
                if len(self.gaze_history) > 5:
                    self.gaze_history.pop(0)
                gaze_x, gaze_y = np.mean(self.gaze_history, axis=0)
                frame = self.draw_holographic_ui(frame, gaze_x, gaze_y)
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            frame = self.process_frame(frame)
            cv2.imshow("Neural Interface", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandsFreeController()
    controller.run()