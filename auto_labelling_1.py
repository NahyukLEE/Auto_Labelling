import os
from os import listdir
from os.path import isfile, join
import natsort

import cv2 
import numpy as np

'''
# object class definition
while True :
    class_name = input("Input your class name :")
    
    if class_name == 'refrigerator':
        class_num = 0
        break
    elif class_name == 'cooker':
        class_num = 1
        break
    else :
        print("Error : you can input only 'refrigerator' or 'cooker'\n")
'''

# Image Load
img_folder =        r'C:\Users\CGlab\Desktop\auto\images'
label_save_folder = r'C:\Users\CGlab\Desktop\auto\labels'
os.chdir(img_folder)

img_files = [f for f in listdir(img_folder) if isfile(join(img_folder, f))]
img_files = natsort.natsorted(img_files, reverse=True)

img_files_num = len(img_files)
count = 0

# Set Minimum Feature Number
MIN_MATCH_COUNT = 10

# Set Minimum Bounding Box's Area Similarity
MIN_AREA_SIMILARITY = 60


# Crop Image (Setting Bounding Box)
query_img = cv2.imread(img_files[0], cv2.IMREAD_GRAYSCALE)
query_img = cv2.resize(query_img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR )

fromCenter = False
r = cv2.selectROI(query_img, fromCenter)
img1 = query_img[int(r[1]):int(r[1]+r[3]),int(r[0]):int(r[0]+r[2])]
print("----------------------------------")


for img_name in img_files:

    h,w = img1.shape
    
    os.chdir(img_folder)
    
    img2 = cv2.imread(img_name,0)
    #img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR) ###
    print("Loading", img_name)
    
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
 
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
 
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 500)
 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
 
    matches = flann.knnMatch(des1,des2,k=2)
 
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 100 * n.distance :
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
        
        print("before : ", x_list)
        print("before : ", y_list)
        
        for i in range (0, 3, 1):
            if x_list[i] <= 0 :
                x_list[i] = 0
            if y_list[i] <= 0 :
                y_list[i] = 0
        
        print("after : ", x_list)
        print("after : ", y_list)
        
        
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
        '''
        cv2.imshow('img', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()
        '''
        img2 = cv2.imread(img_name,0)
        #img2 = cv2.resize(img2, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR ) ###
        
        img1 = img2[y_list[0]:y_list[0] + ( y_list[3] - y_list[0] ), x_list[0]:x_list[0] + ( x_list[3] - x_list[0] )]
        
        
        os.chdir(label_save_folder)
        
        name = os.path.splitext(img_name)
        name = os.path.split(name[0])
        
        label = open(name[1]+".txt", "w")
        label.write(str(class_num) + ' ' + str(round(x,6)) + ' ' + str(round(y,6)) + ' ' + str(round(w,6)) + ' ' + str(round(h,6)))
        
        print("----------------------------------")
        
        
        
    else :
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    
    
    
    if (cv2.waitKey(1) & 0xFF) == 27 : #esc
        cv2.destroyAllWindows()
        break

 
