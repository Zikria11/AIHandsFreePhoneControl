import cv2
import mediapipe as mp

class HandsFreeController:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
        self.cap = cv2.VideoCapture(0)  # Webcam (0 is default)

    def process_frame(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye landmarks (left: 33, right: 263)
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]
                # Print gaze coordinates
                print(f"Left Eye: ({left_eye.x:.2f}, {left_eye.y:.2f}), Right Eye: ({right_eye.x:.2f}, {right_eye.y:.2f})")
                # Draw circles on eyes for visual feedback
                h, w, _ = frame.shape
                cv2.circle(frame, (int(left_eye.x * w), int(left_eye.y * h)), 5, (0, 255, 0), -1)
                cv2.circle(frame, (int(right_eye.x * w), int(right_eye.y * h)), 5, (0, 255, 0), -1)
        return frame

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            frame = self.process_frame(frame)
            cv2.imshow("Hands-Free Control", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
                break
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = HandsFreeController()
    controller.run()