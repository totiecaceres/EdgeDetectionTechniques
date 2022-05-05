import cv2
import numpy as np
import os
import glob
img_dir = "C:\\Users\\Josephine\\Pictures\\Histogram Images\\Images"
os.chdir(img_dir)
idx = 0
name = ['Adnu01','Adnu01_A','Adnu01_B','Adnu01_C','Adnu01_D']

for f1 in glob.glob('*.jpg'):

    img = cv2.imread(f1,0)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Gray " + name[idx] + ".jpg",img)
    #Sobel Operation
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\GRAY.jpg",img)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobelx = np.uint8(np.absolute(sobelx))
    sobely = np.uint8(np.absolute(sobely))

    sobelcombined = cv2.bitwise_or(sobelx,sobely)

    sobelx2 = cv2.Sobel(img,cv2.CV_8U,2,0)
    sobely2 = cv2.Sobel(img,cv2.CV_8U,0,2)
    sobelx2 = np.uint8(np.absolute(sobelx2))
    sobely2 = np.uint8(np.absolute(sobely2))

    sobelcombined2 = cv2.bitwise_or(sobelx2,sobely2)

    # cv2.imshow("sobel x",sobelx)
    # cv2.imshow("sobel y",sobely)
    cv2.imshow("Sobel Combined "+name[idx],sobelcombined)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Sobel_Combined "+name[idx]+".jpg",sobelcombined)
    cv2.imshow("Sobel Combined2 "+name[idx],sobelcombined2)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Sobel_Combined2 "+name[idx]+".jpg",sobelcombined2)
    # cv2.waitKey()

    #Laplace Operation

    lap = cv2.Laplacian(img, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    lap2 = cv2.Laplacian(img, cv2.CV_8U)
    lap2 = np.uint8(np.absolute(lap2))
    lap3 = cv2.Laplacian(img, cv2.CV_16S)
    lap3 = np.uint8(np.absolute(lap3))
    cv2.imshow("Laplacian 64F "+name[idx],lap)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Laplacian_64F "+name[idx]+".jpg",lap)
    cv2.imshow("Laplacian 8U "+name[idx],lap2)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Laplacian_8U "+name[idx]+".jpg",lap2)
    cv2.imshow("Laplacian 16S "+name[idx],lap3)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Laplacian_16S "+name[idx]+".jpg",lap3)
    # cv2.waitKey()

    # Canny Operation

    canny = cv2.Canny(img,30,150)
    cv2.imshow("Canny "+name[idx], canny)
    cv2.imwrite("C:\\Users\\Josephine\\Documents\\Caceres  Act4\\Edge detection\\Canny "+name[idx]+".jpg",canny)
    idx=idx+1
cv2.waitKey()