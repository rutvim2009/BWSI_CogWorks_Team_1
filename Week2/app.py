import streamlit as st
from webcolors import names
from facerecognizer import *
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#redefine show_live_feed to use streamlit
def show_live_feed_streamlit(port=0):
    cap = cv2.VideoCapture(port, cv2.CAP_DSHOW)

    if not cap.isOpened():
        st.error("Could not open camera")
        return

    stframe = st.empty()
    stop = st.button("Stop Live Feed", key="stop_button")

    frame_count = 0

    # Cache these so they aren't recalculated every frame
    profiles = list(db.values())
    threshold = cosine_threshold(db, sample_size=5)

    # Store previous recognition results
    last_boxes = []
    last_names = []
    last_distances = []

    while not stop:
        ok, frame_bgr = cap.read()

        if not ok:
            st.error("Failed to grab frame")
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Only run FaceNet every 5 frames
        if frame_count % 5 == 0:

            boxes, probs, landmarks = model.detect(frame_rgb)

            last_boxes = []
            last_names = []
            last_distances = []

            if boxes is not None and len(boxes) > 0:

                descriptors = model.compute_descriptors(
                    frame_rgb,
                    boxes
                )

                for box, descriptor in zip(boxes, descriptors):
                    name, distance = identify_face(
                        descriptor,
                        profiles,
                        threshold
                    )

                    last_boxes.append(box)
                    last_names.append(name)
                    last_distances.append(distance)

        # Draw the most recent results every frame
        for box, name, distance in zip(
            last_boxes,
            last_names,
            last_distances
        ):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(
                frame_bgr,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            confidence = round(1 - distance, 3)

            cv2.putText(
                frame_bgr,
                name,
                (x1, y1 - 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Draw confidence underneath
            cv2.putText(
                frame_bgr,
                f"Confidence: {confidence}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        display_frame = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        stframe.image(
            display_frame,
            channels="RGB"
        )

    cap.release()

@st.cache_resource
def load_model():
    return FacenetModel()

model = load_model()
with open('face_database.pkl', 'rb') as file:
    db = pickle.load(file)

st.title("Face recognizer")
if st.button("Start Live Feed"):
    show_live_feed_streamlit(port=0)
uploaded_file = st.file_uploader(
    "Please upload an image to recognize a person",
    type = ["jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    
    profiles = list(db.values())
    threshold = cosine_threshold(db, sample_size=5)

   
    
    if st.button("Predict"):
        rgb_image = jpg_to_rgb(image=np.array(image))
        boxes, probs, landmarks= model.detect(rgb_image)
        if boxes is None:
            st.error("Unknown person. Please try again!")
 
        else:
            descriptor = model.compute_descriptors(
                rgb_image,
                boxes
            )[0]

        name, distance = identify_face(
            descriptor,
            profiles,
            threshold
        )
        if name == "Unkown":
            st.error("Unknown person")
        
        else: 
            st.success("Prediction complete")
            st.write("Name: ", name)
            st.write("Confidence: ", str(round((1 - distance),3)))
            matches = find_top_4_sim_faces(
                query_descriptor=descriptor,
                database_profiles=profiles,
                query_name= "Spencer_Hower",
                k=4
            )

            st.write("### Top Similar Matches")
            cols = st.columns(4) # Creates 4 side-by-side columns

            for i, match in enumerate(reversed(matches)):
                with cols[i]:
                    st.write(f"**{match['name']}**")
                    st.write(f"Dist: {match['distance']:.3f}")
                    # If an image path exists in the match dictionary, display it
                    #print(match.get('image'))
                    #if match.get('image') and os.path.exists(match['image']):
                    #    st.image(match["image"], use_container_width=True)
                    #else:
                    #    st.caption("No image available")

        