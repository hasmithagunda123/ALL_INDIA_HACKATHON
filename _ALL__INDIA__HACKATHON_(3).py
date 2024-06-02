import cv2
import numpy as np
from keras.models import load_model
from collections import deque
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import geocoder

# Load the trained model
try:
    model = load_model('model.h5')
except Exception as e:
    print(f"Error loading model: {e}")
IMG_SIZE = 128

# Load Haar cascade classifier for human detection
try:
    human_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
except Exception as e:
    print(f"Error loading Haar cascade: {e}")

class CentroidTracker:
    def __init__(self):
        self.nextObjectID = 0
        self.objects = {}

    def update(self, rects):
        centroids = []

        for rect in rects:
            x, y, w, h = rect
            cx = x + w // 2
            cy = y + h // 2
            centroids.append((cx, cy))

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())

        D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis, :] - np.array(centroids)[np.newaxis, :, :], axis=2)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        usedRows = set()
        usedCols = set()

        for (row, col) in zip(rows, cols):
            if row in usedRows or col in usedCols:
                continue

            objectID = objectIDs[row]
            self.objects[objectID] = centroids[col]
            usedRows.add(row)
            usedCols.add(col)

        unusedRows = set(range(0, D.shape[0])).difference(usedRows)
        unusedCols = set(range(0, D.shape[1])).difference(usedCols)

        for row in unusedRows:
            objectID = objectIDs[row]
            self.objects.pop(objectID)

        for col in unusedCols:
            self.objects[self.nextObjectID] = centroids[col]
            self.nextObjectID += 1

        return self.objects

# Function to send email alert
def send_email(subject, body, to_email, location_info):
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    smtp_username = "your_email@gmail.com"
    smtp_password = "your_password"

    message = MIMEMultipart()
    message["From"] = smtp_username
    message["To"] = to_email
    message["Subject"] = subject

    body_with_location = f"{body}\n\nLocation (Latitude, Longitude): {location_info}"
    message.attach(MIMEText(body_with_location, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_username, to_email, message.as_string())
        print("Email sent successfully!")
        server.quit()
    except smtplib.SMTPException as e:
        print(f"Error sending email: {e}")

# Function to detect humans in the frame
def detect_humans(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return humans

# Function to detect violence in real-time video stream
def detect_violence(video_stream):
    vs = cv2.VideoCapture(video_stream)
    Q = deque(maxlen=128)

    ct = CentroidTracker()

    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("Error grabbing frame")
            break

        try:
            humans = detect_humans(frame)

            if len(humans) > 0:
                objects = ct.update(humans)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                output = cv2.resize(frame, (512, 360)).copy()
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE)).astype("float32") / 255
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)

                results = np.array(Q).mean(axis=0)
                label = results > 0.5
                text = "Violence Detected!" if label else "No Violence"
                color = (0, 0, 255) if label else (0, 255, 0)

                for (objectID, centroid) in objects.items():
                    cv2.putText(output, "ID {}".format(objectID), (centroid[0] - 10, centroid[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.circle(output, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

                cv2.imshow("Output", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                if label:
                    send_alert()

        except Exception as e:
            print(f"Error processing frame: {e}")

    vs.release()
    cv2.destroyAllWindows()

# Function to get current location
def get_current_location():
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng
    else:
        return None

# Function to send alert
def send_alert():
    subject = "Violence Detected!"
    body = "Violence has been detected in the video stream. Please take necessary actions."
    to_email = "recipient@example.com"

    location = get_current_location()
    if location:
        location_info = f"Latitude: {location[0]}, Longitude: {location[1]}"
    else:
        location_info = "Failed to retrieve location."

    send_email(subject, body, to_email, location_info)

# Main function to run real-time violence detection
def main():
    video_stream = 0
    detect_violence(video_stream)

if __name__ == "__main__":
    main()
