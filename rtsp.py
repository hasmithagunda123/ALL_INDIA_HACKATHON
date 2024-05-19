import cv2
import argparse

parser = argparse.ArgumentParser("-r", "--rtsp", "rtsp streaming url")

args = parser.parse_args()


cap = cv2.VideoCapture(args.rtsp)


while(cap.isOpened):
    suc,frame= cap.open()
    
    if not suc:
        print("Ending the frame as no more frames are available")

        break

    ## Further Code Goes Here

import cv2

rtsp_url = "rtsp://username:password@ip_address:554/path"

cap = cv2.VideoCapture(rtsp_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame from RTSP stream")
        break
    
    cv2.imshow("RTSP Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
