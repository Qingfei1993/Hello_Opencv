import cv2
import numpy as np


def empty(x):
    pass


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


class opencv():
    def __init__(self, img):
        self.img = img
        self.img2 = img

    def fuzzy_img(self, kernelsize=(7, 7), sigmaX=0.1, dst=None, sigmaY=None, borderType=None):
        cv2.namedWindow('fuzzy')
        cv2.resizeWindow('fuzzy', 600, 150)
        cv2.createTrackbar('kernelsize', 'fuzzy', 1, 10, empty)
        cv2.createTrackbar('sigmaX', 'fuzzy', 0, 5, empty)
        while True:
            kernel = cv2.getTrackbarPos('kernelsize', 'fuzzy')
            sigmaX = cv2.getTrackbarPos('sigmaX', 'fuzzy')
            img_g = cv2.GaussianBlur(self.img, (2*kernel+1, 2*kernel+1), sigmaX, dst, sigmaY, borderType)
            cv2.imshow('output_img', img_g)
            if cv2.waitKey(10) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
        flag = input('Is it saved? (yes/no)')
        if flag == 'yes': self.img = img_g
        else: pass

    def canny_detect(self):
        pass

    def img_mask(self):
        pass

    def face_detect(self):
        face_detect_img = self.img.copy()
        imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        Cascade_detect = cv2.CascadeClassifier('mode/haarcascade_frontalface_default.xml')
        faces = Cascade_detect.detectMultiScale(imgGray, 1.1, 4)
        for face in faces:
            x, y, w, h = face
            cv2.rectangle(face_detect_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.imshow('face_detect', stackImages(1, ([self.img, face_detect_img])))
        cv2.waitKey(0)
        cv2.destroyWindow('face_detect')
        pass

    def polygon_detect(self):
        imgGray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('polygon_detect')
        cv2.resizeWindow('polygon_detect', 600, 200)
        cv2.createTrackbar('kernelsize', 'polygon_detect', 1, 10, empty)
        cv2.createTrackbar('sigmaX', 'polygon_detect', 0, 5, empty)
        cv2.createTrackbar('thershold1', 'polygon_detect', 0, 255, empty)
        cv2.createTrackbar('thershold2', 'polygon_detect', 255, 255, empty)
        while True:
            kernel = cv2.getTrackbarPos('kernelsize', 'polygon_detect')
            sigmaX = cv2.getTrackbarPos('sigmaX', 'polygon_detect')
            thershold1 = cv2.getTrackbarPos('thershold1', 'polygon_detect')
            thershold2 = cv2.getTrackbarPos('thershold2', 'polygon_detect')
            imgGauss = cv2.GaussianBlur(imgGray, (2*kernel+1, 2*kernel+1), sigmaX)
            imgCanny = cv2.Canny(imgGauss, thershold1, thershold2)
            cv2.imshow('output_img', stackImages(1, ([self.img, imgCanny])))
            if cv2.waitKey(10) & 0xff == ord('q'):
                cv2.destroyAllWindows()
                break
        flag = input('Is it saved to img:2? (yes or no)')
        if flag == 'yes': self.img2 = imgCanny
        else: pass
        img_polygen_detect = self.img.copy()
        flag2 = input('Whether to polygon_detect? (yse or no)')
        if flag2 == 'yes': img_polygen_detect = self.getcounts(imgCanny, img_polygen_detect)
        cv2.imshow('polygon_detect_result', stackImages(1, ([self.img, imgGauss], [imgCanny, img_polygen_detect])))
        cv2.waitKey(0)
        cv2.destroyWindow('polygon_detect_result')

    def getcounts(self, imgCanny, img_polygen_detect):
        contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for contour in contours:
            Area = cv2.contourArea(contour)
            if Area > 500:
                cv2.drawContours(img_polygen_detect, contour, -1, (255, 0, 0), 3)
                length = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.02*length, True)
                vertices_num = len(approx)
                x, y, w, h = cv2.boundingRect(approx)

                if vertices_num == 3: objecttype = 'triangle'
                elif vertices_num == 4:
                    rate = w/h
                    if (rate > 0.95) and (rate < 1.05): objecttype = 'square'
                    else: objecttype = 'rectangle'
                else: objecttype = 'none'
                cv2.rectangle(img_polygen_detect, (x, y), (x+w, y+h), (0, 255, 0), 3)
                cv2.putText(img_polygen_detect, objecttype, (x+w+10, y+h+10), cv2.FONT_HERSHEY_COMPLEX, 0.07, (0, 0, 0))
        return img_polygen_detect


    def show(self, img=None):
        if img == 2: img = self.img2
        else: img = self.img
        cv2.namedWindow('output_img')
        cv2.imshow('output_img', img)
        if cv2.waitKey(0) & 0xff == ord('q'):
            cv2.destroyWindow('output_img')

