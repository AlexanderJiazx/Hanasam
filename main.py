import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import cv2
import time
from landmarks import draw_landmarks_on_image
from pynput.mouse import Controller, Button
import tkinter as tk  # For obtaining screen size
from collections import deque

# get screen size
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# the padding
padding_x = 0.15  
padding_y_top = 0.5
padding_y_bottom = 0.01  

# compute the safe area
x_min = padding_x
x_max = 1 - padding_x
y_min = padding_y_top
y_max = 1 - padding_y_bottom


latest_result = None
clicking = False  # State variable to track whether we are currently clicking
touching = False
pressing = False
previous_z = None  # To track previous z for relative movement
first_press = time.perf_counter()
right_touching = False
right_first_press = time.perf_counter()

STABLIZER_LEVEL = 5
#Stablizer
stablizer_array = deque(maxlen=STABLIZER_LEVEL)


# Initialize pynput mouse controller
mouse = Controller()

# Callback function to handle detection results
def result_callback(result: vision.HandLandmarkerResult, 
                    output_image: mp.Image, 
                    timestamp_ms: int):
    global latest_result
    latest_result = result

base_options = python.BaseOptions(model_asset_path='model.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=result_callback)

detector = vision.HandLandmarker.create_from_options(options)

# set up video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('Hand Landmarks', cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    # mirror
    frame_bgr = cv2.flip(frame_bgr, 1)

    # covert to RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Create a MediaPipe Image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Get timestamp
    timestamp_ms = int(time.time() * 1000)

    # Process the image asynchronously
    detector.detect_async(mp_image, timestamp_ms)

    # If we have a detection result, draw it
    if latest_result:
        # Draw landmarks on the RGB frame
        annotated_frame_rgb = draw_landmarks_on_image(frame_rgb, latest_result)

        # Convert back to BGR for display
        annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)

        # Draw the safe area on the frame for visualization
        frame_height, frame_width, _ = frame_bgr.shape
        top_left = (int(x_min * frame_width), int(y_min * frame_height))
        bottom_right = (int(x_max * frame_width), int(y_max * frame_height))
        cv2.rectangle(annotated_frame_bgr, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(annotated_frame_bgr, "Safe Area", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display landmark coordinates and process cursor movement and clicking
        if latest_result.hand_landmarks:
            for hand_landmarks in latest_result.hand_landmarks:
                # Wrist landmark (0)
                position = hand_landmarks[0]
                stablizer_array.append([position.x,position.y])
                if len(stablizer_array) != stablizer_array.maxlen:
                    x_norm = position.x
                    y_norm = position.y
                else:
                    xSum = 0
                    ySum = 0
                    for pair in stablizer_array:
                        xSum = xSum + pair[0]
                        ySum = ySum + pair[1]
                    x_norm = xSum / len(stablizer_array)
                    y_norm = ySum / len(stablizer_array)

                # Adjust for the safe area
                if x_min <= x_norm <= x_max and y_min <= y_norm <= y_max:
                    # Normalize within the safe area
                    x_safe_norm = (x_norm - x_min) / (x_max - x_min)
                    y_safe_norm = (y_norm - y_min) / (y_max - y_min)

                    # Map to screen coordinates
                    x_screen = int(x_safe_norm * screen_width)
                    y_screen = int(y_safe_norm * screen_height)
                    
                    # Debug: Print screen coordinates
                    print(f"Screen Coordinates: x={x_screen}, y={y_screen}")

                    # Move the cursor using pynput
                    mouse.position = (x_screen, y_screen)
                else:
                    # Optionally, handle cases where the hand is outside the safe area
                    print("Hand is outside the safe area.")

                # Gesture recognition for left click
                thumb_tip = hand_landmarks[4]
                index_tip = hand_landmarks[8]
                middle_finger_tip = hand_landmarks[12]

                # compute the distance between fingers
                dx = thumb_tip.x - index_tip.x
                dy = thumb_tip.y - index_tip.y
                distance = np.hypot(dx, dy)

                ex = thumb_tip.x - middle_finger_tip.x
                ey = thumb_tip.y - middle_finger_tip.y
                right_click_distance = np.hypot(ex, ey)



                # the threshold of touching
                click_threshold = 0.05  # Adjust this value as needed
                right_key_click_threshold = 0.05

                # Calculate press or click
                print("touching status"+str(touching))
                if distance < click_threshold:
                    print("touched, calculating")
                    if not touching:
                            touching = True
                            first_press = time.perf_counter()
                    else:
                        if (time.perf_counter() - first_press) * 1000 < 100:
                            print("touched, but not long enough")
                        #pressed, but not long enough
                        else:
                            print("pressing!!!")
                            if pressing:
                                pass
                            else:
                                mouse.press(Button.left)
                                pressing = True
                                print("Mouse Down")
                    
                        
                elif distance >= click_threshold and touching:
                    # touching, but now fingers are apart
                    touching = False
                    if pressing:
                        mouse.release(Button.left)
                        print("Mouse Up")
                        pressing = False
                    else:
                        #Should click
                        mouse.click(Button.left, 1)
                        print("clicked")
                    
                
                ##################
                # Right Key Part #
                 ################

                if right_click_distance < right_key_click_threshold:
                    if not right_touching:
                        right_touching = True
                        right_first_press = time.perf_counter()

                
                    

                elif right_click_distance >= right_key_click_threshold and right_touching:

                    right_touching = False
                    if (time.perf_counter() - right_first_press) * 1000 < 1000:
                        mouse.click(Button.right,1)


                #  Display landmark coordinates on the frame
                for i, landmark in enumerate(hand_landmarks):
                    x_pixel = int(landmark.x * frame_bgr.shape[1])
                    y_pixel = int(landmark.y * frame_bgr.shape[0])
                    cv2.putText(annotated_frame_bgr, 
                                f"{i}: ({x_pixel}, {y_pixel}, {landmark.z:.2f})", 
                                (10, 30 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                1)


                # Example: Calculate relative z between wrist (0) and index finger tip (8)
                index_tip_z = index_tip.z
                relative_z = index_tip_z - position.z  # Positive if index tip is closer to camera

                # Display the relative z-value on the frame
                cv2.putText(annotated_frame_bgr, 
                            f"Relative Z (Index - Wrist): {relative_z:.3f}", 
                            (10, frame_height - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            (0, 0, 255), 
                            2)

                if relative_z > 0.05:
                    color = (0, 255, 0)  # Green for moving closer
                elif relative_z < -0.05:
                    color = (255, 0, 0)  # Blue for moving away
                else:
                    color = (0, 255, 255)  # Yellow for neutral

                cv2.putText(annotated_frame_bgr, 
                            f"Depth: {'Closer' if relative_z > 0 else 'Farther' if relative_z < 0 else 'Neutral'}", 
                            (10, frame_height - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.7, 
                            color, 
                            2)

        # Display the annotated frame
        cv2.imshow('Hand Landmarks', annotated_frame_bgr)
    else:
        # If no detection result yet, show the original frame
        cv2.imshow('Hand Landmarks', frame_bgr)

    # Exit logic
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Ensure mouse button is released if still presed
        if pressing:
            mouse.release(Button.left)
        break

# Release resources
cap.release()
cv2.destroyAllWindows()