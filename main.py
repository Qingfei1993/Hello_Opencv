import cv2
import numpy as np
from class_opencv_f import opencv


# img_arc = input("请输入图片名称包括后缀")
img_arc = 'img\lena.png'
img = cv2.imread(img_arc)
func_lists = ['show', 'fuzzy_img', 'polygon_detect', 'face_detect']


if __name__ == '__main__':
    raw = opencv(img)
    while True:
        func = input('please input operation that you want to do:')
        if func.split('(')[0].strip() == 'q':
            break
        elif func.split('(')[0].strip() not in func_lists:
            print('your operation is error')
        else:
            eval('raw.' + func)
