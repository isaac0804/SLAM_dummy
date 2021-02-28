import cv2


class Extractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None

    def extract(self, image):
        # detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(image=gray, maxCorners=1000, qualityLevel=0.01, minDistance=3)

        # extraction
        kps = [cv2.KeyPoint(x=corner[0][0], y=corner[0][1], _size=20) for corner in corners]
        kps, des = self.orb.compute(image=image, keypoints=kps)

        # matching
        ret = []
        if self.last is not None:
            # knnMatch returns a list of k(=2) best matches
            matches = self.bf.knnMatch(des, self.last['des'], k=2)
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    ret.append((kps[m.queryIdx], self.last['kps'][m.trainIdx]))
        self.last = {"kps": kps, "des": des}

        # return
        return ret


'''
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
    corners = cv2.goodFeaturesToTrack(gray, 1500, 0.01, 3)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        img[y - 1:y + 1, x - 1:x + 1] = [0, 0, 255]
    cv2.imshow('frame', img)


# ORB (Oriented FAST and Rotated BRIEF)
def orb_method(img):
    orb = cv2.ORB_create()
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    cv2.imshow('frame', img)
'''
