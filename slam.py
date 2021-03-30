#! /usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from frame import Frame, denormalize, match_frames
from pointmap import Point, Map
import PIL.Image
import PIL.ImageOps

# By guessing, focal length, f
# 500 for driving.mp4
# 230 for driving2.mp4
# 1000 for driving_timelapse.mp4
F = int(os.getenv("F", "230"))

# By guessing, focal length, f
# ? for driving.mp4
# 230 for driving2.mp4
# 1000 for driving_timelapse.mp4
F = 230

# Main classes
mapp = Map()


def exif_transpose(img):
    if not img:
        return img

    exif_orientation_tag = 274

    # Check for EXIF data (only present on some files)
    if hasattr(img, "_getexif") and isinstance(img._getexif(), dict) and exif_orientation_tag in img._getexif():
        exif_data = img._getexif()
        orientation = exif_data[exif_orientation_tag]

        # Handle EXIF Orientation
        if orientation == 1:
            # Normal image - nothing to do!
            pass
        elif orientation == 2:
            # Mirrored left to right
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180)
        elif orientation == 4:
            # Mirrored top to bottom
            img = img.rotate(180).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 5:
            # Mirrored along top-left diagonal
            img = img.rotate(-90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 6:
            # Rotated 90 degrees
            img = img.rotate(-90, expand=True)
        elif orientation == 7:
            # Mirrored along top-right diagonal
            img = img.rotate(90, expand=True).transpose(PIL.Image.FLIP_LEFT_RIGHT)
        elif orientation == 8:
            # Rotated 270 degrees
            img = img.rotate(90, expand=True)

    return img


def triangulate(pose1, pose2, pts1, pts2):
    # we only need the upper matrix (3x4 matrix) to triangulate
    # return cv2.triangulatePoints(pose1[:3], pose2[:3], pts1.T, pts2.T).T
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret


def processing_frame(image):
    # Camera intrinsics (Intrinsic Matrix includes information about camera focal length)
    H, W = image.shape[0] // 2, image.shape[1] // 2
    K = np.array([[F, 0, W // 2], [0, F, H // 2], [0, 0, 1]])
    # Resize frame and turn to grayscale
    frame_resized = cv2.resize(image, (W, H))
    gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
    frame = Frame(mapp, gray, K)
    if frame.id == 0:
        return
    print(f"***** frame {frame.id} *****")
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]
    idx1, idx2, Rt = match_frames(f1, f2)
    # idx1 is a list contains valid indices of points in current frame
    # idx2 is a list contains indices of corresponding points (of current frame) from the previous frame
    # to get the coordinates of the valid points, we simply put f1.kps[idx1], for instance

    f1.pose = np.dot(Rt, f2.pose)  # update the new pose (a 4x4 matrix)

    # search in previous frame to see if there is same points
    for i, idx in enumerate(idx2):
        # None indicates the point detected in current frame is new, vice versa
        if f2.pts[idx] is not None:
            f2.pts[idx].add_observation(f1, idx1[i])

    # homogeneous 3D coords
    pts4d = triangulate(f1.pose, f2.pose, f1.kps[idx1], f2.kps[idx2])
    pts4d /= pts4d[:, 3:]
    # length of pts4d is the same as the index of idx1 (because pts4d is just converted 3d coordinates)\

    unmatched_points = np.array([f1.pts[i] is None for i in idx1])
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0) & unmatched_points
    print(f"Adding {sum(good_pts4d)} points")

    print(len(good_pts4d), sum(good_pts4d))
    for i, p in enumerate(pts4d):
        if not good_pts4d[i]:
            continue
        u, v = int(round(f1.kpus[idx1[i], 0])), int(round(f1.kpus[idx1[i], 1]))
        pt = Point(mapp, p, frame_resized[v, u])
        pt.add_observation(f1, idx1[i])
        pt.add_observation(f2, idx2[i])

    for pt1, pt2 in zip(f1.kps[idx1], f2.kps[idx2]):
        # denormalize coordinate (current frame)
        u1, v1 = denormalize(pt1, K)
        # denormalize coordinate (last frame)
        u2, v2 = denormalize(pt2, K)

        cv2.circle(frame_resized, (u2, v2), color=(0, 0, 255), radius=3)
        cv2.circle(frame_resized, (u1, v1), color=(0, 255, 0), radius=3)
        frame_resized = cv2.line(frame_resized, (u1, v1), (u2, v2), color=(255, 0, 0))

    cv2.imshow("frame", frame_resized)
    if frame.id > 4:
        mapp.optimize()
    mapp.display()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("%s <video.mp4>" % sys.argv[0])
        exit(-1)
    cap = cv2.VideoCapture(sys.argv[1])
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
            # frame = exif_transpose(frame)
            processing_frame(frame)

    cap.release()
    cv2.destroyAllWindows()
