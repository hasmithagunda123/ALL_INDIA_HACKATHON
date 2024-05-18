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
    