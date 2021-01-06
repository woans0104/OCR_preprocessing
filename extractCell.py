import sys, os
import numpy as np
import cv2
from sys import argv
from pathlib import Path
import time


MARGIN_W = 15
MARGIN_H = 15
DEBUG = True
# 칼라 양식지 여부 판단 변수
IS_COLORED_FORMAT = False

# 이미지의 평균 칼라값(R, G, B)의 합이 675 이상이면 Contrast 적용
CONTRAST_COLOR_BASE = 675

# 이미지의 평균 색깔 합
COLOR_AVG_SUM = 0

# 파란색 계열의 픽셀 수가 10개 이상이면 칼라 양식지로 판단함
COLORED_PIXEL_COUNT = 10

DIRPATH = ""
FILENAME = ""
EXTENSION = ""

LOG_INDEX = 0


########################################################################################################################################
#@route('/cvt/<fileName>/<type>')
def index(fileName):
    return auto_scan_image(FILENAME)

def showIamge(title, img):
    if DEBUG:
        global DIRPATH, FILENAME, EXTENSION, LOG_INDEX
        LOG_INDEX += 1
        cv2.imwrite(os.path.join(DIRPATH, FILENAME + '_' + str(LOG_INDEX) + '_' + title + EXTENSION), img)
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, img)
        cv2.waitKey()

def auto_scan_image(img):
    global DIRPATH, FILENAME, EXTENSION
    DIRPATH = os.path.dirname(img)
    temp = os.path.basename(img)
    FILENAME = os.path.splitext(temp)[0]
    EXTENSION = os.path.splitext(temp)[1]
    
    try:
        image = cv2.imread(img)
        original = image.copy()
    except Exception as e:
        print(f"> {FILENAME} : {e}")
        print(f"> {FILENAME} : {-1}")
        return img
        
    # 1. 가장 큰 외곽의 사각형을 찾아서 바로 세우기 --------------------------------------------------
    r = 1000 / image.shape[0]
    dim = (int(image.shape[1] * r), 1000)
    image = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 40, 200)
    
    
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
    original = setUpright(original, cnts[0], r)
    t = 1000 / original.shape[1]
    dim = (1000, int(original.shape[0] * t))
    original = cv2.resize(original, dim, interpolation = cv2.INTER_AREA)
    contrasted = img_Contrast(original)
    haha = original.copy()
    mm = original.copy()
    
    global COLOR_AVG_SUM
    COLOR_AVG_SUM = getColorAvgSum(original)
    print(f"COLOR_AVG_SUM : {COLOR_AVG_SUM}")

    # 2-1. 평균 칼라 값이 기준치 이상으로 밝은 경우 Contrast 상향 조정 ---------------------------------
    if COLOR_AVG_SUM > CONTRAST_COLOR_BASE:
        original = setContrastUp(original)

    # 4. 이미지 이진화 ------------------------------------------------------------------------
    original = toBinary(original)
    contrasted = toBinary(contrasted)
    # --------------------------------------------------------------------------------------
    
    # 5. 이미지 내 표(직선) 없애기 ---------------------------------------------------------------
    haha = coloringLine(contrasted, haha)
    ht, wh, _ = haha.shape
    blank = np.zeros((ht,wh,3), np.uint8)
    ww = toBinary(haha)
    ww = cv2.bitwise_not(ww)
    kernel = np.ones((10, 10), np.uint8)
    ww = cv2.morphologyEx(ww, cv2.MORPH_CLOSE, kernel)
    cnts= cv2.findContours(
        ww, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    uu = cv2.drawContours(blank, cnts[0], -1, (0, 255, 0), 1)
    
    cell_li, img_line = detect_cell(cnts[0],mm)
    os.makedirs(DIRPATH + "/crop/" + FILENAME)
    
    result = save_cropCell(cell_li,original,DIRPATH + "/crop/" + FILENAME + "/")
    
    cv2.imwrite(os.path.join(DIRPATH, FILENAME + '_useDetectCell' + EXTENSION), img_line)
    cv2.imwrite(os.path.join(DIRPATH, FILENAME + '_boxbox' + EXTENSION), uu)
    
    return FILENAME + '_scanSuccess' + EXTENSION

def img_Contrast(img):
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)
    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))
    # -----Converting image from LAB Color model to RGB model--------------------
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

    
def isColoredFormat(img):

    height, width, channels = img.shape
    size = height * width

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_RANGE, UPPER_RANGE)
    color = cv2.countNonZero(mask)
    print(f"> Color Count : {color}")

    if color >= COLORED_PIXEL_COUNT:
        return True

    return False
    
def getColorAvgSum(img):
    avg_colors = np.average(np.average(img, axis=0), axis=0)

    avg = int(avg_colors[0]) + int(avg_colors[1]) + int(avg_colors[2])
    print(f"Average Color : {avg} = {int(avg_colors[0])} + {int(avg_colors[1])} + {int(avg_colors[2])}")

    return avg

def setContrastUp(input_img, brightness=-64, contrast=64):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def bluring(img):
    kernel = np.ones((5, 5), np.float32)/25
    blur = cv2.filter2D(img, -1, kernel)
    return blur

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def deNoising(orig):
    thresh = cv2.threshold(orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        if cv2.contourArea(c) < 4:
            cv2.drawContours(thresh, [c], -1, (0, 0, 0), -1)

    
    return ~thresh

def coloringLine(orig, colo):
    he, wid = orig.shape
    
    for i in range(0, he):
        chk = 0
        for j in range (0, wid):
            if orig[i, j] == 0:
                colo[i, j] = [255, 255, 255]
                chk = chk + 1
            else:
                colo[i, j] = [255, 255, 255]
                if chk >= 25:
                    for t in range(j - chk, j):
                        colo[i, t] = [0, 0, 255]
                chk = 0
    for i in range(0, wid):
        chk = 0
        for j in range (0, he):
            if orig[j, i] == 0:
                chk = chk + 1
            else:
                if chk >= 28:
                    for t in range(j - chk, j):
                        colo[t, i] = [0, 0, 255]
                chk = 0

    return colo

def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1)*np.dot(d2, d2)))

def setUpright(image, screenCnt, r):

    peri = cv2.arcLength(screenCnt, True)
    approx = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)

    if len(approx) < 4:
        return image


    s = sorted(approx, key = lambda x : (x[0][0] + x[0][1]))
    s1 = sorted(approx, key = lambda x : (x[0][1] - x[0][0]))
    print(f"> sorted approx : {s} {len(s)}")

    n = np.zeros((4, 1, 2), dtype=np.int64)

    n[0] = s[0]
    n[1] = s1[0]
    n[2] = s[len(s)-1]
    n[3] = s1[len(s1)-1]

    print(f"> n : {n[0]}, {n[1]}, {n[2]}, {n[3]}")
    t1 = np.sqrt(np.square(n[2][0][0] - n[0][0][0]) + np.square(n[2][0][1] - n[0][0][1]))
    t2 = np.sqrt(np.square(n[3][0][0] - n[1][0][0]) + np.square(n[3][0][1] - n[1][0][1]))
    if np.abs(t1 - t2) > 50:
        print(f"대각선 안맞음 {t1} {t2}")
        return image

    rect = order_points(n.reshape(4, 2) / r)
    
    for i in range (len(rect)):
        
        
        if i %3 == 0:
            rect[i][0] = rect[i][0] - MARGIN_H
        else:
            rect[i][0] = rect[i][0] + MARGIN_H
  
        
        if i < 2:
            rect[i][-1] = rect[i][-1] - MARGIN_H
        else:
            rect[i][-1] = rect[i][-1] + MARGIN_H
    
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    print(f"rect {topLeft} {topRight} {bottomRight} {bottomLeft}")

    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth  = max([w1, w2])
    maxHeight = max([h1, h2])
    
    dst = np.float32([[0, 0],
                      [maxWidth-1, 0],
                      [maxWidth-1, maxHeight-1],
                      [0, maxHeight-1]])
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def toBinary(image):
    orig = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    if IS_COLORED_FORMAT == False:
        orig = cv2.adaptiveThreshold(orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    else:
        orig = cv2.adaptiveThreshold(orig, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 30)

    #showIamge('toBinary', orig)
    
    orig = unsharp_mask(orig)

    return orig

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)

    #showIamge('unsharp_mask', sharpened)

    return sharpened
    
def deTabling(orig):
    thresh = cv2.threshold(orig, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=4)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(orig, [c], -1, (255, 255, 255), 3)

    # Remove vertical
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=4)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(orig, [c], -1, (255, 255, 255), 3)

    return orig


def detect_cell(cnts,img):
    
    image_copy = img.copy()
   
    im_w,im_h,_ = image_copy.shape
    img_area = im_w * im_h
    
    cell_li = []
    #box_li = []
    
    for cnt in cnts:
    
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        

        
        #if area > img_area * 0.8:
        #    continue
        
        if h > 10 and w > 10:
        
            
            rect = cv2.minAreaRect(cnt)
            box_point = cv2.boxPoints(rect)
            point = order_points(box_point)
            
            # 삐뚤어진 이미지 제거
            #if 8 < abs(point[0][1] - point[1][1]):
            #    print(f"> {cnt} : 삐뚫어짐")
            #    continue
            
            if len(box_point) <4 :
                print(f"> {cnt} : 4개 미만")
                print(len(box_point))
                continue
                
            cell_li.append([x,y,w,h])
            cv2.drawContours(image_copy,[box_point.astype(int)],0,(255,0,0),1)

    return cell_li , image_copy

def save_cropCell(cell,ori_img,save_path):
    
    
    
    save_path = os.path.join(save_path, 'IMG_')
        
    w,h = ori_img.shape
    
    mask = np.zeros((w, h))
    
    count=0
    for x,y,w,h in cell:
        
        count = count+1
        mask[y: y + h, x: x + w] = 255
        
        cv2.imwrite(save_path+str(count)+".jpg", ori_img[y: y + h, x: x + w])
        
    
    mask = (mask*1).astype('uint8')
    masked = cv2.bitwise_and(ori_img,ori_img, mask =mask)
    
    return masked

t, img = argv

if __name__ == '__main__':
    start = time.time()
    auto_scan_image(img)
    print("time :", time.time() - start)
