import streamlit as st
import cv2
import tempfile
import numpy as np
import argparse
import pickle
import cv2
import os
import time
from keras.models import load_model
from collections import deque
import matplotlib.pyplot as plt
import pygame

def check_violence_in_video(video_path):
        # Initialize pygame
        pygame.init()
        pygame.mixer.init()

        # Load siren sound
        # siren_sound = pygame.mixer.Sound('police-6007.mp3')

        # Global variable to indicate if tampering is detected
        tampering_detected = False

        def detect_tampering(frame, fgbg, kernel):
            global tampering_detected

            fgmask = fgbg.apply(frame)
            fgmask = cv2.erode(fgmask, kernel, iterations=5)
            fgmask = cv2.dilate(fgmask, kernel, iterations=5)
            contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            total_area = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w >= 40 or h >= 40:
                    total_area += w * h
            
            if total_area >= int(frame.shape[0]) * int(frame.shape[1]) / 3:
                tampering_detected = True
                cv2.putText(frame, "TAMPERING DETECTED", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                # Play siren sound if not paused
                # if not paused:
                #     siren_sound.play()
                return True
            return False

        def print_results(video, limit=None):
            global tampering_detected

            fig = plt.figure(figsize=(16, 30))
            if not os.path.exists('output'):
                os.mkdir('output')

            print("Loading model ...")
            model = load_model('./model.h5')
            Q = deque(maxlen=128)

            vs = cv2.VideoCapture(video)
            writer = None
            (W, H) = (None, None)
            count = 0

            fgbg = cv2.createBackgroundSubtractorMOG2()
            kernel = np.ones((5, 5), np.uint8)

            while True:
                (grabbed, frame) = vs.read()
                ID = vs.get(1)
                if not grabbed:
                    break
                try:
                    if (ID % 7 == 0):
                        count = count + 1
                        if W is None or H is None:
                            (H, W) = frame.shape[:2]

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        output = cv2.resize(frame, (512, 360)).copy()
                        frame_resized = cv2.resize(frame, (128, 128)).astype("float32")
                        frame_resized = frame_resized.reshape(128, 128, 3) / 255
                        preds = model.predict(np.expand_dims(frame_resized, axis=0))[0]
                        Q.append(preds)

                        results = np.array(Q).mean(axis=0)
                        i = (preds > 0.56)[0]  # np.argmax(results)
                        label = i
                        text = "Violence: {}".format(label)
                        
                        color = (0, 255, 0)
                        if label:
                            color = (255, 0, 0)
                        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                        # Check for tampering
                        if detect_tampering(frame, fgbg, kernel):
                            function("Tampering")
                            break

                        # saving mp4 with labels but cv2.imshow is not working with this notebook
                        if writer is None:
                            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                            writer = cv2.VideoWriter("output.mp4", fourcc, 60, (W, H), True)

                        writer.write(output)
                        
                        fig.add_subplot(8, 3, count)
                        plt.imshow(output)

                    if limit and count > limit:
                        break

                except Exception as e:
                    print(f"Exception occurred: {e}")
                    break

            plt.show()
            print("Cleaning up...")
            if writer is not None:
                writer.release()
            vs.release()
            if tampering_detected:
                function("Tampering")
            else:
                if f:
                    function("Violence")


st.set_page_config(page_title="Violence Detection in Video", page_icon=":movie_camera:")

st.title("🎥 Violence Detection in Video")

uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    st.video(uploaded_file)

    st.write("⏳ Processing video... Please wait.")
    
    # Call your function to check for violence
    violence_detected = check_violence_in_video(tmp_file_path)

    if violence_detected:
        st.error("❌ Violence detected in the video.")
    else:
        st.success("✅ No violence detected in the video.")
