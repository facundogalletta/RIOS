
import numpy as np
import cv2

def calibration_dji(img,height,width):
    
    img=np.array(img)
    # loaded_data = np.load('D:/fgalletta/calibration_inspire2/calibracion.npz')
    # mtx = loaded_data['mtx']
    # dist = loaded_data['dist']
    # rvecs = loaded_data['rvecs']
    # tvecs = loaded_data['tvecs']
    
    # undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    # undistorted_img=np.array(undistorted_img)
    
    # print(undistorted_img.shape)
    return img
