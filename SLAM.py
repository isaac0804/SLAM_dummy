import cv2
import numpy as np

cap = cv2.VideoCapture('Videos/driving2.mp4')

H = 1920 // 2
W = 1080 // 2


# Harris Corner Detection
def hc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.03 * dst.max()] = [0, 0, 255]
    cv2.imshow('frame', img)


# Shi-Tomasi Corner Detection
def stc(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 1500, 0.07, 3)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        img[y - 1:y + 1, x - 1:x + 1] = [0, 0, 255]
    cv2.imshow('frame', img)


while cap.isOpened():
    ret, frame = cap.read()
    # HC for Harris
    # STC for Shi-Tomasi
    feature_detection = "HC"
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
    # Resize frame
    frame_resized = cv2.resize(frame, (H, W))
    # Choose feature detection type
    if feature_detection == "HC":
        hc(frame_resized)
    elif feature_detection == "STC":
        stc(frame_resized)

cap.release()
cv2.destroyAllWindows()
