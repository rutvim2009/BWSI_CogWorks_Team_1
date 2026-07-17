from facerecognizer import *
from whispers import *
from facenet_models import FacenetModel

model = FacenetModel()
with open('face_database.pkl', 'rb') as file:
    db = pickle.load(file)

dataset_path = "Image_Dataset/lfw-funneled/lfw_funneled"

"""for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue
    
    add_profile(db, person)

    for filename in os.listdir(person_path)[:5]:# can change [:5] to limit to first 5 images for testing
        if filename.endswith(".jpg"):
            image_path = os.path.join(person_path, filename)
            img = jpg_to_rgb(image_path=image_path)
            add_images(db, person, img, model)

print(f"Loaded {len(db)} people into the dataset.")

import pickle

with open("face_database.pkl", "wb") as f:
    pickle.dump(db, f)

print("Database saved!")"""

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

        boxes, probs, landmarks = model.detect(frame_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            descriptor = model.compute_descriptors(frame_rgb, boxes)[0]
            profiles = list(db.values())
            threshold = cosine_threshold(db, sample_size=5)
            name, distance = identify_face(descriptor, profiles, threshold)
            cv2.putText(
                frame_bgr,
                name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame_bgr,
            "Confidence: " + str(round((1 - distance),3)),
                (x1, y1 + 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
        cv2.imshow("Live Feed", frame_bgr)
        # Wait 1ms between frames; break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_live_feed(port=0)