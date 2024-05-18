import cv2
import numpy as np

def detect_and_highlight_moving_objects(video_path, output_path):
    """
    Detect and highlight moving objects in a video using Optical Flow.

    Parameters:
    video_path (str): Path to the input video file.
    output_path (str): Path to save the output video with moving objects highlighted.
    """
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(first_frame)

    # Define the color for highlighting moving objects (BGR format)
    color = (0, 255, 0)  # Green color

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and direction of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold magnitude to identify motion
        mask_motion = magnitude > 5  # Adjust threshold as needed

        # Highlight moving objects
        mask[mask_motion] = color

        # Display the resulting frame
        result = cv2.add(frame, mask)

        # Write the resulting frame to the output video
        cv2.imwrite(output_path, result)

        # Update previous frame
        prev_gray = gray

    # Release video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = '/content/drive/MyDrive/archieve/SCVD/videos/Non-Violence Videos/nv110.mp4'
output_path = '/content/drive/MyDrive/archieve/SCVD/moving_objects_highlighted.mp4'
detect_and_highlight_moving_objects(video_path, output_path)
