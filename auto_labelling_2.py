import os
from os import listdir
from os.path import isfile, join
import natsort

import cv2
import numpy as np

class_num = 0 

# Image Load
img_folder = r'C:\Users\CGlab\Desktop\auto\images'
label_save_folder = r'C:\Users\CGlab\Desktop\auto\labels'
os.chdir(img_folder)

img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
img_files = natsort.natsorted(img_files, reverse=True)

img_files_num = len(img_files)
count = 0

# Set Minimum Feature Number
MIN_MATCH_COUNT = 10

query_img_01 = r'C:\Users\CGlab\Desktop\auto\images\frame0.jpg' # 전방 우
query_img_02 = r'C:\Users\CGlab\Desktop\auto\images\frame132.jpg' # 전방 좌
query_img_03 = r'C:\Users\CGlab\Desktop\auto\images\frame254.jpg' # 후방 우
query_img_04 =r'C:\Users\CGlab\Desktop\auto\images\frame414.jpg' # 후방 좌

fromCenter = False

query_img = cv2.imread(query_img_01, cv2.IMREAD_GRAYSCALE)
r = cv2.selectROI(query_img, fromCenter)
q_img1 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
cv2.destroyAllWindows()

query_img = cv2.imread(query_img_02, cv2.IMREAD_GRAYSCALE)
r = cv2.selectROI(query_img, fromCenter)
q_img2 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
cv2.destroyAllWindows()

query_img = cv2.imread(query_img_03, cv2.IMREAD_GRAYSCALE)
r = cv2.selectROI(query_img, fromCenter)
q_img3 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
cv2.destroyAllWindows()

query_img = cv2.imread(query_img_04, cv2.IMREAD_GRAYSCALE)
r = cv2.selectROI(query_img, fromCenter)
q_img4 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
cv2.destroyAllWindows()

img_arr = [q_img1, q_img2, q_img3, q_img4]

cv2.imshow('img',img_arr[0])
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow('img',img_arr[1])
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow('img',img_arr[2])
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imshow('img',img_arr[3])
cv2.waitKey()
cv2.destroyAllWindows()


for img_name in img_files:
    
    os.chdir(img_folder)
    
    img2 = cv2.imread(img_name,0)
    print("Loading", img_name)
    
    img_value = 0
    
    for i in range (0, 4, 1) :
    
        h,w = img_arr[i].shape
        
        k = i + 1
        
        print("Loading %d/4 ... " % k)
        
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
     
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_arr[i],None)
        kp2, des2 = sift.detectAndCompute(img2,None)
     
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
     
        flann = cv2.FlannBasedMatcher(index_params, search_params)
     
        matches = flann.knnMatch(des1,des2,k=2)
     
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 10 * n.distance :
                good.append(m)

        #print("Match Points :", len(good), "(> 10)")
        
        if len(good) >= img_value :
            img_value = len(good)
            img1 = img_arr[i]
    
    
    h,w = img1.shape
    
    sift = cv2.xfeatures2d.SIFT_create()
 
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
 
    matches = flann.knnMatch(des1,des2,k=2)
 
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 10 * n.distance :
            good.append(m)
    
    print("Match Points :", len(good), "(> 10)")
    
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
     
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
     
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
     
        
        cv2.polylines(img2,[np.int32(dst)],True,(0,255,0),2) 
        
        
        
        x_list = [np.int32(dst)[0][0][0], np.int32(dst)[1][0][0], np.int32(dst)[2][0][0], np.int32(dst)[3][0][0]]
        x_list.sort()
        y_list = [np.int32(dst)[0][0][1], np.int32(dst)[1][0][1], np.int32(dst)[2][0][1], np.int32(dst)[3][0][1]]
        y_list.sort()
        
        #print("before : ", x_list)
        #print("before : ", y_list)
        
        for i in range (0, 3, 1):
            if x_list[i] <= 0 :
                x_list[i] = 0
            if y_list[i] <= 0 :
                y_list[i] = 0
        
        #print("after : ", x_list)
        #print("after : ", y_list)
        
        
        # New Label
        cv2.rectangle(img2, (x_list[0], y_list[0]), (x_list[3], y_list[3]), (0,255,0), 1)
        
        
        w = ( x_list[3] - x_list[0] ) / img2.shape[1]
        h = ( y_list[3] - y_list[0] ) / img2.shape[0]
        x = ((x_list[0] + x_list[3]) / 2 ) /img2.shape[1]
        y = ((y_list[0] + y_list[3]) / 2 ) /img2.shape[0]
        print("%d %.6f %.6f %.6f %.6f" % (class_num, x , y, w, h))
        
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                           singlePointColor = None,
                           matchesMask = matchesMask, # draw only inliers
                           flags = 2)
        
            
        img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        
        
        cv2.imshow('img', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        
        img2 = cv2.imread(img_name,0)
        #img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR ) ###
        
        #img1 = img2[y_list[0]:y_list[0] + ( y_list[3] - y_list[0] ), x_list[0]:x_list[0] + ( x_list[3] - x_list[0] )]
        
        
        os.chdir(label_save_folder)
        
        name = os.path.splitext(img_name)
        name = os.path.split(name[0])
        
        label = open(name[1]+".txt", "w")
        label.write(str(class_num) + ' ' + str(round(x,6)) + ' ' + str(round(y,6)) + ' ' + str(round(w,6)) + ' ' + str(round(h,6)))
        
        print("Save", name[1]+".txt")
        print("----------------------------------")
        
        
    else :
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    count += 1
    
    if (cv2.waitKey(1) & 0xFF) == 27 : #esc
        cv2.destroyAllWindows()
        break
    

    
    
    