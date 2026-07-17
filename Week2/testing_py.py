import cv2
import numpy as np

def show_live_feed(port=0, backend=cv2.CAP_DSHOW):
    cap = cv2.VideoCapture(port, backend)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at port {port}")

    print("Press 'q' to quit.")

    while True:
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            print("Failed to grab frame")
            break
        
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cv2.imshow("Live Feed", frame_bgr)

        # Wait 1ms between frames; break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_live_feed(port=0)