#! /usr/bin/env python3
import cv2
import numpy as np
from frame import Frame, denormalize, match_frames, IRt
from dotmap import Point, Map

# Camera intrinsics (Intrinsic Matrix includes information about camera focal length)
W, H = 1920 // 2, 1080 // 2
F = 230  # By guessing, focal length, f = 230 for driving2.mp4
K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])

# Main classes
mapp = Map()


def triangulate(pose1, pose2, pts1, pts2):
    # we only need the upper matrix (3x4 matrix) to triangulate
    return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T


def processing_frame(image):
    # Resize frame and turn to grayscale
    frame_resized = cv2.resize(image, (W, H))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

    frame = Frame(mapp, gray, K)
    if frame.id == 0:
        return

    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    # idx1 is a list contains valid indices of points in current frame
    # idx2 is a list contains indices of corresponding points (of current frame) from the previous frame
    # to get the coordinates of the valid points, we simply put f1.pts[idx1], for instance

    f1.pose = np.dot(Rt, f2.pose)  # update the new pose (a 4x4 matrix)

    # homogeneous 3D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])
    pts4d /= pts4d[:, 3:]
    # length of pts4d is the same as the index of idx1 (because pts4d is just converted 3d coordinates)

    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:,2] > 0)

    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        # denormalize coordinate (current frame)
        u1, v1 = denormalize(pt1, K)
        # denormalize coordinate (last frame)
        u2, v2 = denormalize(pt2, K)

        cv2.circle(frame_resized, (u2, v2), color=(0, 0, 255), radius=3)
        cv2.circle(frame_resized, (u1, v1), color=(0, 255, 0), radius=3)
        frame_resized = cv2.line(frame_resized, (u1, v1), (u2, v2), color=(255, 0, 0))

    cv2.imshow("frame", frame_resized)
    mapp.display()


if __name__ == "__main__":
    cap = cv2.VideoCapture('Videos/driving2.mp4')
    while cap.isOpened():
        ret, frame = cap.read()

        # cv2.waitKey(x) waits for x milliseconds and returns an integer value based on the key input.
        # However, we only want the last byte (8 bits) of it to prevent potential bug(activation of NumLock for instance).
        # 0xFF is a hexadecimal constant 11111111 in binary.
        # AND (&) is a bitwise operator, purpose here is to keep the last byte.
        # ord('') returns the ASCII value of the character which would be again maximum 255.
        # REMEMBER to press the desired key on the pop up window not terminal.
        # If the video ends, frame will be None, so we have to put the while loop before the frame resized.

        if cv2.waitKey(1) & 0xFF == ord('q') or frame is None:
            break
        else:
            processing_frame(frame)

    cap.release()
    cv2.destroyAllWindows()
