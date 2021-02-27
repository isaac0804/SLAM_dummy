import cv2
import numpy as np

cap = cv2.VideoCapture('Videos/driving.mp4')

H = 1920 // 2
W = 1080 // 2

while cap.isOpened():
    ret, frame = cap.read()
    '''
    - cv2.waitKey(x) waits for x milliseconds and returns an integer value based on the key input. However, we only want 
    the last byte (8 bits) of it to prevent potential bug(activation of NumLock for instance).
    - 0xFF is a hexadecimal constant 11111111 in binary.
    - AND (&) is a bitwise operator, purpose here is to keep the last byte.
    - ord('') returns the ASCII value of the character which would be again maximum 255.
    - REMEMBER to press the desired key on the pop up window not terminal.
    - If the video ends, frame will be None, so we have to put the while loop before the frame resized.
    '''
    if cv2.waitKey(1) & 0xFF == ord('q') or frame is None:
        break
    frame_resized = cv2.resize(frame, (H, W))
    cv2.imshow('frame', frame_resized)

cap.release()
cv2.destroyAllWindows()
