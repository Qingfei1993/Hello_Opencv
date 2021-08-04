import cv2
import numpy as np

def empty(x):
    pass

class opencv():
    def __init__(self, img):
        self.img = img

    def fuzzy_img(self, kernelsize=(7, 7), sigmaX=0.1, dst=None, sigmaY=None, borderType=None):
        cv2.namedWindow('fuzzy')
        cv2.resizeWindow('fuzzy', 600, 300)
        cv2.createTrackbar('kernelsize', 'fuzzy', 1, 9, empty)
        cv2.createTrackbar('sigmaX', 'fuzzy', 1, 5, empty)
        while True:
            kernel = cv2.getTrackbarPos('kernelsize', 'fuzzy')
            sigmaX = cv2.getTrackbarPos('sigmaX', 'fuzzy')
            print(kernel, sigmaX)
            img_g = cv2.GaussianBlur(self.img, (kernel, kernel), sigmaX, dst, sigmaY, borderType)
            print('--')
            cv2.imshow('output_img', img_g)
            print('---')
            if cv2.waitKey(10) & 0xff == ord('q'):
                break
        pass

    def canny_detect(self):
        pass

    def img_mask(self):
        pass

    def face_detect(self):
        pass

    def polygon_detect(self):
        pass

    def show(self):
        cv2.namedWindow('output_img')
        cv2.imshow('output_img', self.img)
        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyWindow('output_img')
