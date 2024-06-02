import cv2
import numpy as np

def detect_and_highlight_moving_objects(video_path, output_path):
    """
    Detect and highlight moving objects in a video using Optical Flow.

    Parameters:
    video_path (str): Path to the input video file.
    output_path (str): Path to save the output video with moving objects highlighted.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return


    # Convert the first frame to grayscale
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Create an empty mask with the same shape as the first frame
    mask = np.zeros_like(first_frame)

    # Define the color for highlighting moving objects (BGR format)
    color = (0, 255, 0)  # Green color

    while True:
        # Read the next frame from the video
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate Optical Flow between the previous and current grayscale frames
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Compute magnitude and direction of flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Threshold magnitude to identify motion
        mask_motion = magnitude > 5  # Adjust threshold as needed

        # Highlight moving objects by setting the mask to the defined color where motion is detected
        mask[mask_motion] = color

        # Display the resulting frame with the mask to highlight moving objects
        result = cv2.add(frame, mask)

        # Write the resulting frame to the output video
        cv2.imwrite(output_path, result)

        # Update previous frame with the mask to highlight moving objects
        prev_gray = gray

        # Release video capture and close all OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

        # Example usage
        video_path = '/content/drive/MyDrive/archieve/SCVD/videos/Non-Violence Videos/nv110.mp4'
        output_path = '/content/drive/MyDrive/archieve/SCVD/moving_objects_highlighted.mp4'
        detect_and_highlight_moving_objects(video_path, output_path)
