import cv2
import numpy as np
from scipy.spatial import ConvexHull
from skimage import filters as fl
import pyhdust.images as phim

class NucleusSegmentation:
    def __init__(self, min_area=100):
        self.min_area = min_area

    def segmentation(self, img):
        org_img = img.copy()

        # Color balancing
        Gray = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]

        mean_gray = np.mean(Gray)
        mean_R = np.mean(R)
        mean_G = np.mean(G)
        mean_B = np.mean(B)

        R_ = R * (mean_gray / mean_R)
        G_ = G * (mean_gray / mean_G)
        B_ = B * (mean_gray / mean_B)

        R_[R_ > 255] = 255
        G_[G_ > 255] = 255
        B_[B_ > 255] = 255

        balance_img = np.zeros_like(org_img)
        balance_img[:, :, 0] = R_.copy()
        balance_img[:, :, 1] = G_.copy()
        balance_img[:, :, 2] = B_.copy()

        cmyk = phim.rgb2cmyk(balance_img)
        _M = cmyk[:, :, 1]
        _K = cmyk[:, :, 3]

        _S = cv2.cvtColor(balance_img, cv2.COLOR_RGB2HLS_FULL)[:, :, 2]

        min_MS = np.minimum(_M, _S)
        a_temp = np.where(_K < _M, _K, _M)
        KM = _K - a_temp

        b_temp = np.where(min_MS < KM, min_MS, KM)
        min_MS_KM = min_MS - b_temp

        min_MS_KM = cv2.GaussianBlur(min_MS_KM, ksize=(5, 5), sigmaX=0)
        try:
            thresh2 = fl.threshold_multiotsu(min_MS_KM, 2)
            Nucleus_img = np.zeros_like(min_MS_KM)
            Nucleus_img[min_MS_KM >= thresh2] = 255
        except:
            _M = cv2.GaussianBlur(_M, ksize=(5, 5), sigmaX=0)
            thresh2 = fl.threshold_multiotsu(_M, 2)
            Nucleus_img = np.zeros_like(_M)
            Nucleus_img[_M >= thresh2] = 255

        contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        pad_del = np.zeros_like(Nucleus_img)

        max_area = max(cv2.contourArea(contours[idx]) for idx in np.arange(len(contours)))
        for j in range(len(contours)):
            if cv2.contourArea(contours[j]) < (max_area / 10):
                cv2.drawContours(pad_del, contours, j, color=255, thickness=-1)
        Nucleus_img[pad_del > 0] = 0

        contours, _ = cv2.findContours(Nucleus_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _perimeter = 0
        for cnt in contours:
            _perimeter += cv2.arcLength(cnt, True)

        temp_points = np.argwhere(Nucleus_img == 255)
        Ncl_points = np.zeros_like(temp_points)
        Ncl_points[:, 0] = temp_points[:, 1]
        Ncl_points[:, 1] = temp_points[:, 0]
        _area = np.sum(Nucleus_img)

        cvx_hull = ConvexHull(Ncl_points)
        img_convex = np.zeros_like(Nucleus_img)
        cv2.drawContours(img_convex, [cvx_hull.points[cvx_hull.vertices]], 0, color=255, thickness=-1)

        img_ROC = img_convex - Nucleus_img

        return Nucleus_img, img_convex, img_ROC

    def feature_extractor(self, img):
        flag, error, features = self.segmentation(img)
        if not flag:
            print(error)
            return None
        return features
