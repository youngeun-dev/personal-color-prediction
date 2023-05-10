# 최종적으로 모델이 진단한 퍼스널 컬러와 opencv 코드가 진단한 퍼스널 컬러가 일치하는지 확인하기 위한 python 파일
# dataset을 구축하기 위해 작성된 color_classifier.py 파일 안의 함수 및 main 구조가 동일하다

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2 as cv
import warnings
from collections import OrderedDict
import dlib
import imutils
import matplotlib

warnings.filterwarnings(action='ignore') 

def cheek(img) :
    CHEEK_IDXS = OrderedDict([("left_cheek", (1, 2, 3, 4, 5, 48, 31)),
                            ("right_cheek", (11, 12, 13, 14, 15, 35, 54))
                            ])

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    img = imutils.resize(img, width=600)

    overlay = img.copy()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    bounding_rects = []
    detections = detector(gray, 0)
    for k, d in enumerate(detections):
        shape = predictor(gray, d)
        for (_, name) in enumerate(CHEEK_IDXS.keys()):
            pts = np.zeros((len(CHEEK_IDXS[name]), 2), np.int32)
            for i, j in enumerate(CHEEK_IDXS[name]):
                pts[i] = [shape.part(j).x, shape.part(j).y]
            pts = pts.reshape((-1, 1, 2))
            cv.drawContours(overlay, [pts], -1, (255, 255, 255), -1, cv.LINE_AA)
            bounding_rects.append(cv.boundingRect(pts))
    height, width=overlay.shape[:2]

    for y in range(0, height):
        for x in range(0, width):
            b=overlay.item(y, x, 0)
            g=overlay.item(y, x, 1)
            r=overlay.item(y, x, 2)  
            if b==255 and g==255 and r==255:
                continue
            overlay.itemset(y, x, 0, 0)
            overlay.itemset(y, x, 1, 0)
            overlay.itemset(y, x, 2, 0)
    output = 0
    output=cv.bitwise_and(img, overlay, output)
    black=0
    for y in range(0, height):
        for x in range(0, width):
            b=output.item(y, x, 0)
            g=output.item(y, x, 1)
            r=output.item(y, x, 2)
            if b==0 and g==0 and r==0:
                black+=1
            if b==255 and g==255 and r==255:
                output.itemset(y, x, 0, 0)
                output.itemset(y, x, 1, 0)
                output.itemset(y, x, 2, 0)
    if black > height*width-10 : return img
    return output

class Color :
    person_HSV = []
        
    def color_classifier(self, person_HSV) :    
        self.H = float(person_HSV[0])
        self.S = float(person_HSV[1])
        self.V = float(person_HSV[2])
        diff = round(self.V - self.S, 2)
        print(self.H, diff)
        if self.H >= 20 and self.H <= 210 : 
            if diff >= 68.25 :
                    self.ans = 0
                    # Spring
            elif diff < 68.25:
                    self.ans = 1
                    # Autumn
        elif (self.H >= 0 and self.H < 20) or (self.H > 210 and self.H < 360) :
            if diff >= 68.75 :
                    self.ans = 2
                    # Summer
            elif diff < 68.75:
                    self.ans = 3
                    # Winter

        else :
            self.ans = -1
            # 에러
            
        return self.ans

def color_ratio(clt) :
    numLabels = np.arange(0, len(np.unique(clt.labels_))+1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
    return bar

def skin_detector(img, file_name) :
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    converted = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    skinMask = cv.inRange(converted, lower, upper)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))
    skinMask = cv.erode(skinMask, kernel, iterations = 2)
    skinMask = cv.dilate(skinMask, kernel, iterations = 2)

    skinMask = cv.GaussianBlur(skinMask, (3,3), 0)
    skin = cv.bitwise_and(img, img, mask = skinMask)

    result = skin
    img = cv.cvtColor(result, cv.COLOR_BGR2HLS)
    skin_img = img
    temp_img = cv.cvtColor(img, cv.COLOR_HLS2RGB)

    h, w, c = img.shape

    for i in range(h) :
        for j in range(w) :
            H = img[i][j][0]
            L = img[i][j][1]
            S = img[i][j][2]

            R = temp_img[i][j][0]
            G = temp_img[i][j][1]
            B = temp_img[i][j][2]

            if R==0 and G==0 and B==0:
                continue

            LS_ratio = L/S
            skin_pixel = bool((S>=50) and (LS_ratio > 0.5) and (LS_ratio < 3.0) and ((H <= 25) or (H >= 165)))
            temp_pixel = bool((R == G) and (G == B) and (R >= 220))

            if skin_pixel :
                if temp_pixel :
                    skin_img[i][j][0] = 0
                    skin_img[i][j][1] = 0
                    skin_img[i][j][2] = 0
                else :
                    pass
            else :
                skin_img[i][j][0] = 0
                skin_img[i][j][1] = 0
                skin_img[i][j][2] = 0

    skin_img = cv.cvtColor(skin_img, cv.COLOR_HLS2BGR)            
    for i in range(h) :
        for j in range(w) :
            B = skin_img[i][j][0]
            G = skin_img[i][j][1]
            R = skin_img[i][j][2]

            bg_pixel = bool(B==0 and G==0 and R==0)

            if bg_pixel :
                skin_img[i][j][0] = 255
                skin_img[i][j][1] = 255
                skin_img[i][j][2] = 255
            else :
                pass
    
    cvt_img = cv.cvtColor(skin_img, cv.COLOR_BGR2RGB)
    cvt_img = cvt_img.reshape((cvt_img.shape[0]*cvt_img.shape[1], 3))
    k = 20
    clt = KMeans(n_clusters=k)
    clt.fit(cvt_img)
    
    
    hist = color_ratio(clt)
    temp = np.array(clt.cluster_centers_)

    del_index = hist.argmax()
    hist = np.delete(hist, del_index)
    temp = np.delete(temp, del_index, 0)

    
    try :
        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0) 

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)

        del_index = np.argmin(hist)
        hist = np.delete(hist, del_index)
        temp = np.delete(temp, del_index, 0)
    except ValueError :
        print(file_name, "에러")
        pass
    
    hist = hist / hist.sum()
    bar = plot_colors(hist, temp)

    bar = cv.cvtColor(bar, cv.COLOR_BGR2RGB)
    
    return bar

def color_convert2(cheek):
    img=cheek
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    R = []
    G = []
    B = []
    for i in img :
        for j in i :
            R.append(j[0])
            G.append(j[1])
            B.append(j[2])

    R_sum = 0
    G_sum = 0
    B_sum = 0

    # 각 R, G, B의 합계 구하기
    for i in range(len(R)) :
        R_sum += R[i]
        G_sum += G[i]
        B_sum += B[i]

    R_avg = int(round((R_sum / len(R)), 0))  # R값 평균
    G_avg = int(round((G_sum / len(G)), 0))  # G값 평균
    B_avg = int(round((B_sum / len(B)), 0))  # B값 평균

    img_avg = img

    for i in img_avg :
        for j in i :
            j[0] = R_avg
            j[1] = G_avg
            j[2] = B_avg

    bgr_img_avg = cv.cvtColor(img_avg, cv.COLOR_RGB2BGR)
    bgr_img_avg=cv.resize(bgr_img_avg, (50, 50))
    return bgr_img_avg

def color_convert(cheek) :
    img=cheek
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    sum = 0
    R = []
    G = []
    B = []
    for i in img :
        for j in i :
            R.append(j[0])
            G.append(j[1])
            B.append(j[2])

    R_sum = 0
    G_sum = 0
    B_sum = 0

    # 각 R, G, B의 합계 구하기
    for i in range(len(R)) :
        R_sum += R[i]
        G_sum += G[i]
        B_sum += B[i]

    R_avg = int(round((R_sum / len(R)), 0))  # R값 평균
    G_avg = int(round((G_sum / len(G)), 0))  # G값 평균
    B_avg = int(round((B_sum / len(B)), 0))  # B값 평균
    RGB_color = [R_avg, G_avg, B_avg]
           
    arr_RGB_color = np.array(RGB_color)
    float_arr_RGB_color = arr_RGB_color / 255
    float_tp_RGB_color = tuple(float_arr_RGB_color)
    HSV_color = matplotlib.colors.rgb_to_hsv(float_tp_RGB_color)
    HSV_color2 = np.array([round(HSV_color[0]*359, 3), round(HSV_color[1] * 100, 3)-4, round(HSV_color[2] * 100, 3)+8])
    HSV_color2 = list(HSV_color2)
    HSV_color2[0] = round(HSV_color2[0], 2)
    HSV_color2[1] = round(HSV_color2[1], 2)
    HSV_color2[2] = round(HSV_color2[2], 2)
    return HSV_color2

def save_img(file_name, skin_type) :

    if skin_type == 0 :
        print("spring")
    elif skin_type == 1:
        print("autumn")
    elif skin_type == 2:
        print("summer")
    elif skin_type == 3:
        print("winter")
    elif skin_type == -1:
        print("")
        print(file_name, "에러")

img = cv.imread("person.png") # 퍼스널컬러 알고싶은 사진
file_name="personal"
cheekimg = cheek(img)
cv.imwrite("cheek.png", cheekimg)
bar = skin_detector(cheekimg, file_name)
cv.imwrite("bar.png", bar)
bgr = color_convert2(bar)
hsv = color_convert(bar)

color_class = Color()
skin_type = color_class.color_classifier(hsv)
cv.imwrite("skin.png", bgr)
save_img(file_name, skin_type)