import cv2
import numpy as np
from extractor import Extractor

W = 1920 // 2
H = 1080 // 2

extractor = Extractor()


def processing_frame(image):
    # Resize frame
    frame_resized = cv2.resize(image, (W, H))

    # Get Feature Matches from Extractor
    matches = extractor.extract(frame_resized)
    if matches:
        for m in matches:
            # extract coordinate information(current frame) from KeyPoint using .pt
            u1, v1 = map(lambda x: int(round(x)), m[0].pt)
            # extract coordinate information(last frame) from KeyPoint using .pt
            u2, v2 = map(lambda x: int(round(x)), m[1].pt)

            cv2.circle(frame_resized, (u1, v1), color=(0, 255, 0), radius=3)
            cv2.circle(frame_resized, (u2, v2), color=(0, 0, 255), radius=3)
            frame_resized = cv2.line(frame_resized, (u1, v1), (u2, v2), color=(255, 0, 0))

    cv2.imshow("frame", frame_resized)


if __name__ == "__main__":
    cap = cv2.VideoCapture('Videos/driving2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()
        '''
        - cv2.waitKey(x) waits for x milliseconds and returns an integer value based on the key input. However, we only 
        want the last byte (8 bits) of it to prevent potential bug(activation of NumLock for instance).
        - 0xFF is a hexadecimal constant 11111111 in binary.
        - AND (&) is a bitwise operator, purpose here is to keep the last byte.
        - ord('') returns the ASCII value of the character which would be again maximum 255.
        - REMEMBER to press the desired key on the pop up window not terminal.
        - If the video ends, frame will be None, so we have to put the while loop before the frame resized.
        '''
        if cv2.waitKey(1) & 0xFF == ord('q') or frame is None:
            break
        else:
            processing_frame(frame)

    cap.release()
    cv2.destroyAllWindows()
