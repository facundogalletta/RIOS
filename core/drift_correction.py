
from __future__ import print_function, unicode_literals
import cv2 as cv
import numpy as np


from intersection_captures import intersection
from georectification import georef_img
from micasense import capture as capture
from prefix import prefijo
from build_output_paths import build_output_paths
from get_camera_params import get_camera_params
from thermal_processing_setup import thermal_processing_setup
from apply_mask import apply_mask

def drift_correction(intersection_values,correction_drift_method,coef_corr,k,range_im,temp_i,img,imRec,name_0,name,img_type,cam,sz_window,PanelNames,SkyNames,band,FOVwidth,FOVheight,csv_file,DT_vignetting,correction_vignetting):
    if not(range_im[0]==range_im[1]):
        # Correction for secuential images (more than 1 image)
        if k == range_im[0]:
            
            ######################################################################################################################
            ##------------------------------------------------------------------------------------------------------------------##
            ##-- Correcting the fist image with an external known value that coincides with the mean temperature in the image --##
            ##------------------------------------------------------------------------------------------------------------------##
            ######################################################################################################################
            
            # "img" is the Level 1 corrected image
            img       = np.array(img)
            
            m2 = np.mean(img[int(img.shape[0]/2-sz_window[0]/2):int(img.shape[0]/2+sz_window[0]/2),
                            int(img.shape[1]/2-sz_window[1]/2):int(img.shape[1]/2+sz_window[1]/2)]) # Calculate mean value in the window defined
            
            
            if correction_drift_method == 'mult':
                coef_corr.append(temp_i/m2)
                im_res = cv.multiply(np.array(imRec[0]).astype(np.float32),np.float64(coef_corr[-1]))
                I_np2=cv.multiply(np.array(img),np.float64(coef_corr[-1]))
            elif correction_drift_method == 'add':
                coef_corr.append(temp_i-m2)
                im_res = cv.add(np.array(imRec[0]).astype(np.float32),np.float64(coef_corr[-1]))
                I_np2=cv.add(np.array(img),np.float64(coef_corr[-1]))
            
        else:
            
            #################################################################
            ##-------------------------------------------------------------##
            ##-- Correcting the secuential image with the previous image --##
            ##-------------------------------------------------------------##
            #################################################################
                            
            # Apply intersection function
            px_1,px_2, px_1r, px_2r,px_bar1,px_bar2,im1Rec,im2Rec,im1,im2 = intersection(img_type, cam, name_0, name, FOVwidth, FOVheight, band, csv_file, flag_features='d',
                                                                                flag_interseccion=False, DT_vignetting=DT_vignetting, correction_vignetting=correction_vignetting,
                                                                                panel_names=PanelNames, sky_names=SkyNames)
            
            # For the previuos image, the correction was determined in the previous step (coef_corr[-1])
            if correction_drift_method == 'mult':
                I_np1 = cv.multiply(np.array(im1),np.float64(coef_corr[-1]))
                im1Rec=cv.multiply(np.array(im1Rec[0]),np.float64(coef_corr[-1]))
            else: 
                I_np1=cv.add(np.array(im1),np.float64(coef_corr[-1]))
                im1Rec=cv.add(np.array(im1Rec[0]),np.float64(coef_corr[-1]))
            
            I_np2=np.array(im2)
            im2Rec=np.array(im2Rec[0])
            
            # Calculating the coeficient of correction with the oblique intersection and with the rectified intersection
            Ir_1_rec=apply_mask(im1Rec,px_1r)
            Ir_2_rec=apply_mask(im2Rec,px_2r)
            
            Ir_1 = apply_mask(I_np1,px_1)
            Ir_2 = apply_mask(I_np2,px_2)
            
            data_1=Ir_1[Ir_1>0]
            data_2=Ir_2[Ir_2>0]
            
            data_1_rec=Ir_1_rec[Ir_1_rec>0]
            data_2_rec=Ir_2_rec[Ir_2_rec>0]
            
            data_filtered_1=cv.GaussianBlur(data_1, (5, 5), 0)
            data_filtered_2=cv.GaussianBlur(data_2, (5, 5), 0)
            
            data_filtered_1_rec=cv.GaussianBlur(data_1_rec, (5, 5), 0)
            data_filtered_2_rec=cv.GaussianBlur(data_2_rec, (5, 5), 0)
            
            # Stats in the intersection region for each image
            
            m1=np.mean(data_filtered_1)
            m2=np.mean(data_filtered_2)
            
            m1_rec=np.mean(data_filtered_1_rec)
            m2_rec=np.mean(data_filtered_2_rec)
            
            if intersection_values=='obl':
                if correction_drift_method == 'mult':
                    coef_corr.append(m1/m2)
                else:
                    coef_corr.append(m1-m2)
            else:
                if correction_drift_method == 'mult':
                    coef_corr.append(m1_rec/m2_rec)
                else:
                    coef_corr.append(m1_rec-m2_rec)
            
            if correction_drift_method == 'mult':
                I_np2  = cv.multiply(np.array(im2),np.float64(coef_corr[-1]))
                im_res = cv.multiply(im2Rec.astype(np.float32),np.float64(coef_corr[-1]))
            else:
                I_np2  = cv.add(np.array(im2),np.float64(coef_corr[-1]))
                im_res = cv.add(im2Rec.astype(np.float32),np.float64(coef_corr[-1]))

            # To zero borders values estapolated in the rectification.
            # im_res[np.where(im_res<temp_i-10)]=0
    else:
        I_np2=np.array(img)
        im_res=imRec[0]
                
    return I_np2,im_res, coef_corr