# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:45:23 2020

@author: user
"""

import cv2
import os
import numpy as np

from dewarping import main


MARGIN_W = 10
MARGIN_H = 10
LOG_INDEX = 0



def findOuter(image,original):
    
    r = 800 / image.shape[0]
    dim = (int(image.shape[1] * r), 800)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 40, 200)
    
    

    kernel = np.ones((2, 2), np.uint8)
    close_img = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


    cnts = cv2.findContours(close_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse=True)
    
    showIamge("contour", cv2.drawContours(image, [cnts[0]], -1, (0, 255, 0), 3))
    
    original = setUpright(original, cnts[0], r)
    
    showIamge("setUpright", original)
    
    return original

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def setUpright(image, screenCnt, r):

    peri = cv2.arcLength(screenCnt, True)
    approx = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)

    if len(approx) < 4:
        return image

    t1 = np.sqrt(np.square(approx[2][0][0] - approx[0][0][0]) + np.square(approx[2][0][1] - approx[0][0][1]))
    t2 = np.sqrt(np.square(approx[3][0][0] - approx[1][0][0]) + np.square(approx[3][0][1] - approx[1][0][1]))
    if np.abs(t1 - t2) > 10:
        return image
    
    s = sorted(approx, key = lambda x : (x[0][0] + x[0][1]))
    
    s1 = sorted(screenCnt, key = lambda x : (x[0][0] - x[0][1]))
    s2 = sorted(screenCnt, key = lambda x : (x[0][1] - x[0][0]))




    print(f"> sorted approx : {s} {len(s)}")

    n = np.zeros((4, 1, 2), dtype=np.int64)

    n[0] = s[0]
    n[1] = s1[-1]
    n[2] = s[-1]
    n[3] = s2[-1]
   

    print(f"> n : {n[0]}, {n[1]}, {n[2]}, {n[3]}")

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
    
    im_w, im_h,_ = image.shape
    w_w,w_h,_  = warped.shape
    
    if (im_w * im_h) * 0.6 > (w_w*w_h) :
        return image
    
    
    return warped

def showIamge(title, img):

    global DIRPATH, FILENAME, EXTENSION, LOG_INDEX
    LOG_INDEX += 1
    cv2.imwrite(os.path.join(DIRPATH, FILENAME + '_' + str(LOG_INDEX) + '_' + title + EXTENSION), img)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey()


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

    #showIamge('setContrastUp', buf)

    return buf

def remove_shadow(image):
    
    
    rgb_planes = cv2.split(image)
    
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)
    
    #result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    
    return result_norm

def box(width, height):
    return np.ones((height, width), dtype=np.uint8)

def find_line(img):
   

    # copy dst, then for horizontal and vertical lines' detection.
    horizontal =img.copy()
    vertical = img.copy()
    
    scale = 20*2  # play with this variable in order to increase/decrease the amount of lines to be detected
    
    # Specify size on horizontal axis
    print(horizontal.shape)
    horizontalsize = horizontal.shape[1] // scale
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    
    
    # vertical
    verticalsize = vertical.shape[0] // scale
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    
    
    
    # table line
    table = horizontal + vertical
    
    return table




def detect_cell(cnts,img):
    
    image_copy = img.copy()
   
    im_w,im_h,_ = image_copy.shape
    img_area = im_w * im_h
    
    cell_li = []
    #box_li = []
    
    for cnt in cnts:
    
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        

        
        if area > img_area * 0.8:
            continue
        
        if area > 200 and h > 15 and w > 5:
        
        # 글자 높이보다는 h가 커야 됨
        # TODO : 평균 글자 높이 잡아서 그거보다 크게 만듬
        
            
            rect = cv2.minAreaRect(cnt)
            box_point = cv2.boxPoints(rect)
            point = order_points(box_point)
            
            # 삐뚤어진 이미지 제거
            if 10 < abs(point[0][1] - point[1][1]):
           
                continue
            
            if len(box_point) <4 :
                
                print(len(box_point))
                continue
                #box = np.int0(box)
                #box_li.append(box)
                
            cell_li.append([x,y,w,h])
            cv2.drawContours(image_copy,[box_point.astype(int)],0,(255,0,0),2)
        
    

    showIamge("drawContours", image_copy)
    
    
    return cell_li , image_copy


def save_cropCell(cell,ori_img,save_path):
    
    
    
    save_path = os.path.join(save_path, 'IMG_')
        
    w,h,_ = ori_img.shape
    
    mask = np.zeros((w, h))
    
    count=0
    for x,y,w,h in cell:
        
        count = count+1
        mask[y: y + h, x: x + w] = 255
        
        cv2.imwrite(save_path+str(count)+".jpg", ori_img[y: y + h, x: x + w])
        
    
    mask = (mask*1).astype('uint8')
    masked = cv2.bitwise_and(ori_img,ori_img, mask =mask)
    
    
    showIamge("setUpright", masked)
    
    
    
    return masked

    
    
    

def main():

    #1.Load image 
    
    img = r'C:\Users\user\Desktop\test\test1.jpg'
    
    #import pdb; pdb.set_trace()
    global DIRPATH, FILENAME, EXTENSION
    
    DIRPATH = os.path.dirname(img)
    DIRPATH = r"C:\Users\user\Desktop\test"
    temp = os.path.basename(img)
    EXTENSION = os.path.splitext(temp)[1]
    FILENAME = os.path.splitext(temp)[0]
    FILENAME = FILENAME + '_output'
    
    
    
    DIRPATH = os.path.join(DIRPATH, FILENAME)
    if not os.path.exists(DIRPATH):
        os.makedirs(DIRPATH)
      
      
    try:
        image = cv2.imread(img)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original = image.copy()
    except Exception as e:
        print(f"> {FILENAME} : {e}")
        print(f"> {FILENAME} : {-1}")


    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,20))
    plt.imshow(original)
    #2.Find the edge of table
    
    for i in range(1):
        original = findOuter(original.copy(),original) 
    
    #3.Resize image
    
    resize_img1 = cv2.resize(original, dsize=(900, 1200), interpolation=cv2.INTER_AREA)


    #4.Remove shadow 
    
    resize_img = remove_shadow(resize_img1)


    #warp_img = main(img,resize_table)


    #5.Contrast increase

    #resize_img = setContrastUp(resize_img)


    #6.rgb -> gray
    
    gray_img = cv2.cvtColor(resize_img, cv2.COLOR_RGB2GRAY)


    #6.1 his equ
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
    cl1 = clahe.apply(gray_img)


    #7.Binarization
    
    binari_img = cv2.adaptiveThreshold(cl1, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY_INV,
                                         55,
                                         7)

    
    #8.Morphology close
    
    kernel = np.ones((3, 3), np.uint8)
    close_img = cv2.morphologyEx(binari_img, cv2.MORPH_CLOSE, kernel)
    
           

    
   
    ################################################
    
    #9.Find line 
    
    table = find_line(close_img)
    
                

    #10.Find contours
    
    cnts= cv2.findContours(
        table, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    
    

    #11.Detect cell
    
    cell_li, img_line = detect_cell(cnts,resize_img)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,20))
    plt.imshow(img_line)
    
    
    #12.image 저장
    
   
    result = save_cropCell(cell_li,resize_img,DIRPATH)
    
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,20))
    plt.imshow(result)
    
    
    
    

    

if __name__ == '__main__':
    main()
    
    







