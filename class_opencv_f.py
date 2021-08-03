import cv2
import numpy as np


class opencv():
    def __init__(self, img):
        self.img = img

    def fuzzy_img(self):
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
