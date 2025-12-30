"""
##################                #########              ############                 ######################
##---------------###              ##-----##           ####----------####            ###----------------###
##----########----###             ##-----##         ###----########----###         ###---------------###  
##----##     ###----###           ##-----##        ###----###    ###----###        ###------##########
##----#       ###----###          ##-----##       ###----###      ###----###       ###-----####
##----##     ###----###           ##-----##      ###----###        ###----###        ###-----####
##----#########----###            ##-----##      ###----###        ###----###          ###-----####      
##---------------###              ##-----##      ###----###        ###----###            ###-----####  
##----######----###               ##-----##      ###----###        ###----###              ###-----####
##----##   ###----###             ##-----##       ###----###      ###----###                  ##-----###
##----##     ###----###           ##-----##        ###----###    ###----###           ###########-----###
##----##       ###----###         ##-----##         ###----########----###          ###---------------###
##----##         ###----###       ##-----##           ####----------####          ###----------------###
########           ###%%%%###     #########              ############           ######################
                                                
Remote sensing with drone Imagery for Observation of water Surfaces

"""

#########################################
##-------------------------------------##
##-- LOAD RIOS ENCIRONMENT LIBRARIES --##
##-------------------------------------##
#########################################

from __future__ import print_function, unicode_literals
import os
import sys
os.environ['PROJ_LIB'] = r'C:/Users/fgalletta/anaconda3/envs/rios_clean/Library/share/proj'
from PIL import Image
import cv2 as cv
import numpy as np
from osgeo import gdal
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import csv
import glob
from shapely.geometry import Polygon, Point      
import progressbar
from time import sleep

import multiprocessing as mp
if __name__ == "__main__":
    mp.freeze_support()
    
########################################
##------------------------------------##
##-- ADD CORE FOLDER TO SYSTEM PATH --##
##------------------------------------##
########################################
  
dir_path = os.path.abspath('../core')
if dir_path not in sys.path:
    sys.path.append(dir_path)

from intersection_captures import intersection
from georectification import georef_img
from micasense import capture as capture
from prefix import prefijo
from build_output_paths import build_output_paths
from get_camera_params import get_camera_params
from thermal_processing_setup import thermal_processing_setup
from apply_mask import apply_mask
from drift_correction import drift_correction


"""
Esta es la version no interactiva (en gran parte) del codigo. 

Se debe acceder la información de:

- SourcePath         : Ruta de las imagenes georeferenciadas
- SkyNames           : Ruta de las imagenes del cielo. None para RIOS-G y RIOS-T
- PanelNames         : Ruta de las imagenes del panel. None para RIOS-G y RIOS-T

- DstPathGeo         : Ruta de las imagenes georeferenciadas
- DstPathRect        : Ruta de las imagenes rectificadas
- csv_file           : Ruta del csv con las coordenadas de las imagenes georeferenciadas - Util cuando los datos gps de la camara no son buenos

- type_process       : Tipo de procesamiento - 2 por defecto en RIOS-T
- cam                : Cámara utilizada
- save               : Guardar las imagenes rectificadas
- [ini,fin]          : Rango de imagenes a procesar
- corregir           : Insdicar si se corrie la deriva de las imagenes termicas
- tipo               : Tipo de corrección de efectos de borde.
- [ini_c,fin_c]      : Rango de imagenes de la termica en agua fría. 
- flag_plot_promedio : Graficar promedio de agua fría utilizado para la corrección de borde.
- temp_i             : Temperatura media de la primera imagen.
- tipo_pendiente     : Tipo de pendiente de corrección de acumulación de errores en la intersección de imagenes.

"""



######################
##------------------##
##-- SOURCE PATHs --##
##------------------##
######################

SourcePath='../examples/DataFly_PuntadelTigre_20211216/'
csv_file = '../examples/DataFly_PuntadelTigre_20211216/csv/flight_record_FLY219.csv'

SkyNames = None
PanelNames = None

#####################################################
##-------------------------------------------------##
##-- Select RIOS module (RIOS-G, RIOS-T, RIOS-R) --##
##-------------------------------------------------##
#####################################################

type_process = '2'       # ['1 - RIOS-G','2 - RIOS-T', 3 - RIOS-R']
cam          = '2'       # ['1 - Micasense Altum (multiespectral)','2 - Micasense Altum (thermal-infrared)','3 - Zenmuse X5S','4 - DJI Mavic Pro','5 - DJI Mini 2','6 - set manual']
band         = 6         # ['1 - Blue','2 - Green','3 - Red','4 - RedEdge','5 - NIR'] - In RIOS-T is 6 and in RIOS-G is 0                       
save         = '2'       # Save rectified (no georeferenced) images ['1 - Yes','2 - No']
[ini,fin]    = [55,225]

########################################
##------------------------------------##
##-- Specificing points of interest --##
##------------------------------------##
########################################

name_points = ['S1','S2','S3','S4','S5','S6','S7','S8']
X_points    = [542676.833  ,542685.372  ,542717.086  ,542725.625  ,542546.06  ,542594.55  ,542807.905  ,542856.395]
Y_points    = [6153551.122 ,6153517.179 ,6153391.107 ,6153357.164 ,6153415.10 ,6153427.29 ,6153480.979 ,6153493.177]

size_pixel_window_points=[3,3,3,3,3,3,3,3] # Size pixel of the window center in each point 
size_meter_window_points=[3,3,3,3,3,3,3,3] # Size meter of the window center in each point 

flag_mask = False # Save a mask window for each point in each image that cointains the point. ['True - Yes','False - No']
                # The mask window corresponds to the window associated with each point used for extract information of each image.

##################################################################
##--------------------------------------------------------------##
##-- Parameteres modifiable for level 2 corrections in RIOS-T --##
##--------------------------------------------------------------##
##################################################################

drift                          = '1'
correction_drift_method        = 'add'                  # Indicate which method to use ['mult' - Miltiply second image for de ratio of mean value images in the intersection, 'add' - Add at the second image the diference of mean value betwen images in the intersection]
intersection_values            =  'obl'                  # Indicate which intersection values to use ['obl' - Pixels in oblique image,'rec' - Pixels in rectified image]
kind_vignetting                = '1'                     # Vignette correction ['1 - Original','2 - Paraboloide','3 - Gaussiana','4 - None']
[ini_c,fin_c]                  = [20,55]                 # Number index of capture range of an object of uniform temperature (i.e. cold water)
range_c                        = [int(ini_c),int(fin_c)] 
flag_plot_average              = '2'                     # Plot average of captures at object of uniform temperature ['1 - Yes','2 - No']
temp_i                         = 22.2                    # Mean temperature in the first image
kind_slope_accumulation        = '3'                     # kind of slope for correction of accumulation errors ['1 - Actual, 2 - Set','3 - None']
actual_slope_accumulation      = -0.013
date_actual_slope_accumulation = '16/12/2020 - 14:00'
DT_vignetting                  = []                      # Initial values of vignetting correction. it will be calculated using the average of captures at object of uniform temperature

##########################
##----------------------##
##-- Set Output Paths --##
##----------------------##
##########################

DstPathGeo, DstPathRect = build_output_paths(SourcePath, type_process, band, ini, fin, drift, kind_vignetting, kind_slope_accumulation,correction_drift_method,intersection_values)

###########################
##-----------------------##
##-- Set Camera Params --##
##-----------------------##
###########################

cam_params  = get_camera_params(type_process, cam)
width       = cam_params['width']
height      = cam_params['height']
FOVwidth    = cam_params['FOVwidth']
FOVheight   = cam_params['FOVheight']
prefix      = cam_params['prefix']
img_type    = cam_params['img_type']

if cam=='3' or cam=='4' or cam=='5':
    SourceImags=glob.glob(os.path.join(SourcePath, '*.JPG'))    
elif cam=='1' or cam=='2':
    SourceImags=glob.glob(os.path.join(SourcePath, '*.tif'))

###############################
##---------------------------##
##-- Creating putput paths --##
##---------------------------##
###############################

if save == '1':
    # If the path exists, clean it
    if os.path.exists(DstPathRect):
        if os.listdir(DstPathRect):
            print(f"Cleaning directory: {DstPathRect}")
            for filename in os.listdir(DstPathRect):
                file_path = os.path.join(DstPathRect, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # borra archivos o enlaces
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # borra subcarpetas
                except Exception as e:
                    print(f'Error deleting {file_path}: {e}')
    else:
        print(f"Creating directory: {DstPathRect}")
        os.makedirs(DstPathRect, exist_ok=True)
    
elif save == '2':
    DstPathRect = ''
    
# If the path exists, clean all
if os.path.exists(DstPathGeo):
    if os.listdir(DstPathGeo):  # si no está vacío
        print(f"Cleaning directory: {DstPathGeo}")
        for filename in os.listdir(DstPathGeo):
            file_path = os.path.join(DstPathGeo, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # borra archivos o enlaces
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # borra subcarpetas
            except Exception as e:
                print(f'Error deleting {file_path}: {e}')
else:
    # If doesn't exists, clean it
    print(f"Creating directory: {DstPathGeo}")
    os.makedirs(DstPathGeo, exist_ok=True)

if cam=='3' or cam=='4' or cam=='5':
    csv_file=[]

NameTXT='Data.txt' 
txtAux='Aux.txt'

##########################
##----------------------##
##-- Processing Range --##
##----------------------##
##########################

range_im=[int(ini),int(fin)]
OutputPathTXT= DstPathGeo+'/'+NameTXT

####################
##----------------##
##-- Processing --##
##----------------##
####################

###################################################
##-----------------------------------------------##
##-- Parameters processing for thermal imagery --##
##-----------------------------------------------##
###################################################

thermal_cfg = thermal_processing_setup(
    img_type=img_type,
    drift=drift,
    kind_vignetting=kind_vignetting,
    kind_slope_accumulation=kind_slope_accumulation,
    SourcePath=SourcePath,
    range_c=range_c,
    band=band,
    actual_slope_accumulation=actual_slope_accumulation,
    flag_plot_average=flag_plot_average
)

temp_mean               = thermal_cfg['temp_mean']
temp_5                  = thermal_cfg['temp_5']
temp_95                 = thermal_cfg['temp_95']
coef_corr               = thermal_cfg['coef_corr']
correction_vignetting   = thermal_cfg['correction_vignetting']
DT_vignetting           = thermal_cfg['DT_vignetting']
slope                   = thermal_cfg['slope']
kind_slope_accumulation = thermal_cfg['kind_slope_accumulation']

# .TXT FROM GDAL       

g=open(DstPathGeo+'/'+txtAux,"w+")
g.write("""RangoFotosi RangoFotosf Band type
"""+str(range_im[0])+' '+str(range_im[-1])+' '+str(band)+ ' ' + str(img_type)+'\n')
g.close()

if cam=='3' or cam=='4' or cam=='5':
    prefix="DJI_"
else:
    prefix="IMG_"
        
# Create Mosaic an structure for each point specified

date_value_points = [[] for _ in range(len(X_points))]

if cam=='3' or cam=='4' or cam=='5':
    elem_mosaico = [[] for _ in range(3)]
    Value_points = [[[] for _ in range(3)] for _ in range(len(X_points))]
else:
    elem_mosaico=[]
    Value_points = [[] for _ in range(len(X_points))]

coef_corr=[]

num_images=range_im[1]-range_im[0]+1
bar = progressbar.ProgressBar(maxval=num_images, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
bar.start()

for k in range(range_im[0],range_im[1]+1):
    
    Prefix=prefijo(k,prefix)
    
    # Define the path of the image and the capture (micasense) of image
    if cam=='3' or cam=='4' or cam=='5':
    
        name=os.path.join(SourcePath, Prefix+str(k)+".JPG")
        capture_img=[]
        flag_features='dji'
    
    elif cam=='1' or cam=='2':
        
        imageNames = glob.glob(os.path.join(SourcePath, Prefix+str(k)+"_*.tif"))
        
        if len(imageNames)==1:
            name=imageNames[0]
        else:
            name=imageNames[band-1]
        
        panel_names = None
        capture_img = capture.Capture.from_filelist(imageNames)
        flag_features='d'
        
    # Load imagel PIL and binary
    
    pil_img = Image.open(name)
    binary_img = open(name,"rb")
    
    ###############################
    ##---------------------------##
    ##-- APPLY GEOREF FUNCTION --##
    ##---------------------------##
    ###############################
    
    res_georef = georef_img(img_type, cam, height, width, FOVwidth, FOVheight, capture_img, pil_img, binary_img, band, 
                            csv_file, flag_features, DT_vignetting, correction_vignetting, PanelNames, SkyNames)
    
    img           = res_georef[0]
    imRec         = res_georef[3]
    height_rec    = res_georef[4]
    width_rec     = res_georef[5]
    resHeight     = res_georef[6]
    resWidth      = res_georef[7]
    pt_upper_left = res_georef[8]
    yawDeg        = res_georef[15]
    date_num      = res_georef[16]
    h_final       = res_georef[17]
    h_final_1     = np.linalg.inv(h_final)
    
    
    ##########################
    ##----------------------##
    ##-- DRIFT CORRECTION --##
    ##----------------------##
    ##########################
    
    if drift == '1':
    
        name_0=os.path.join(SourcePath, prefijo(k-1,prefix)+str(k-1)+"_"+str(band)+".tif")
        sz_window=(30,30)
        I_np2, im_res, coef_corr = drift_correction(intersection_values,correction_drift_method,coef_corr,k,range_im,temp_i,img,imRec,name_0,name,img_type,cam,
                                                    sz_window,PanelNames,SkyNames,band,FOVwidth,FOVheight,csv_file,DT_vignetting,correction_vignetting)
    else:
        
        I_np2=np.array(img)
        im_res=imRec[0]        
    
    x_1=float(pt_upper_left[0])
    y_1=float(pt_upper_left[1])
    
    d=[k,x_1,y_1,resWidth,resHeight,yawDeg]
    
    if k==range_im[0]:
        f=open(OutputPathTXT,"w+")
        f.write(str(k)+' '+str(x_1)+' '+str(y_1)+' '+str(resWidth)+' '+str(resHeight)+' '+str(yawDeg)+ '\n')
        f.close()
    else:
        f=open(OutputPathTXT,"a+")
        f.write(str(k)+' '+str(x_1)+' '+str(y_1)+' '+str(resWidth)+' '+str(resHeight)+' '+str(yawDeg)+ '\n')
        f.close()
    
    if not(DstPathRect==''):
        for i in range(len(imRec)): 
            if img_type==1:
                print(os.path.join(DstPathRect, "orthorectified_"+Prefix+str(k)+"_"+str(i)+".JPG"))
                try:
                    cv.imwrite(DstPathRect+'/'+'orthorectified_'+Prefix+str(k)+"_"+str(i)+".JPG", imRec[i])
                except Exception as e:
                    print("Error al guardar la imagen:", e)
            else:
                cv.imwrite(os.path.join(DstPathRect, "orthorectified_"+Prefix+str(k)+"_"+str(band)+".tif"),im_res)
    

    ###################################################################
    ##---------------------------------------------------------------##
    ##-- Georreferencing images with Geo-transformation parameters --##
    ##---------------------------------------------------------------##
    ###################################################################

    x1     = d[1]
    y1     = d[2]
    resX   = d[3]
    resY   = d[4]
    yawDeg = d[5]
    angle  = yawDeg*np.pi/180
    x_skew = -resY*np.sin(angle)
    y_skew = -resX*np.sin(angle)
    resX   = resX*np.cos(angle)
    resY   = -resY*np.cos(angle)
    
    if img_type==1:
        for i in range(len(imRec)):              
            
            imarray=imRec[i]
            #Dimensiones de la imagen
            [h,b]=imarray.shape
            
            # Create raster for Geotiffs
            fn=DstPathGeo+'/'+Prefix+str(k)+"_"+str(i)+"_Georef_"+".JPG"
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(imarray)

            # Apply geotransform
            geot = [x1, resX, x_skew, y1, y_skew, resY]
            ds.SetGeoTransform(geot)
            srs = gdal.osr.SpatialReference()
            
            # Set projection. The second parameter is: North=1 or South=0
            srs.SetUTM(21,0)
            srs.SetWellKnownGeogCS('WGS84')
            ds.SetProjection(srs.ExportToWkt())
            
            ndv=0#No data value
            ds.GetRasterBand(1).SetNoDataValue(ndv)
            ds = None
            if cam=='3' or cam=='4' or cam=='5':
                elem_mosaico[i].append(fn)
                    
            else:
                elem_mosaico.append(fn)
                
    elif img_type==2:
        
        imarray=np.array(im_res, dtype=np.float32)
        id=np.where(imarray>0)
        id_2=np.where(I_np2>0)
        
        if kind_slope_accumulation == '1' or kind_slope_accumulation =='2':
            imarray[id] = imarray[id] + slope*(k-int(range_im[0]))
            I_np2[id_2] = I_np2[id_2] + slope*(k-int(range_im[0]))
            
        [h,b] = imarray.shape 
        
        # Creating the raster for the geotiff
        
        fn=DstPathGeo+'/'+Prefix+str(k)+"_"+str(band)+"_georeferenced"+".tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(imarray)

        # Apply Geo-transformation parameters
        geot = [x1, resX, x_skew, y1, y_skew, resY]#[500000, 10, 0, 4600000, 0, -10]
        ds.SetGeoTransform(geot)
        srs = gdal.osr.SpatialReference()
        
        ###############################
        ##---------------------------##
        ##-- Define UTM projection --##
        ##---------------------------##
        ###############################
        
        # North=1 ó South=0
        
        srs.SetUTM(21,0)
        srs.SetWellKnownGeogCS('WGS84')
        ds.SetProjection(srs.ExportToWkt())
        
        ndv=0#No data value
        ds.GetRasterBand(1).SetNoDataValue(ndv)
        ds = None
        elem_mosaico.append(fn)
        # eval('elem_mosaico=elem_mosaico+[fn]')
        temp_mean.append(np.mean(I_np2[I_np2>0]))
        #percentiles
        temp_5.append(np.percentile(I_np2[I_np2>0],5))
        temp_95.append(np.percentile(I_np2[I_np2>0],95))
        
        print('Mean temperature in the image ' + str(k) + ' is ' + str(temp_mean[-1]) + '°C')
        
    elif img_type==3:
        
        for i in range(len(imRec)):              
            
            imarray=imRec[i]
            [h,b]=imarray.shape
            
            # Creating the raster for the geotiff
            fn=DstPathGeo+'/'+Prefix+str(k)+"_"+str(i)+"_Georef_"+".JPG"
            driver = gdal.GetDriverByName('GTiff')
            ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
            ds.GetRasterBand(1).WriteArray(imarray)

            # Apply Geo-transformation parameters
            geot = [x1, resX, x_skew, y1, y_skew, resY]
            ds.SetGeoTransform(geot)
            srs = gdal.osr.SpatialReference()
            
            ###############################
            ##---------------------------##
            ##-- Define UTM projection --##
            ##---------------------------##
            ###############################
            
            # North=1 ó South=0
            
            srs.SetUTM(21,0)
            srs.SetWellKnownGeogCS('WGS84')
            ds.SetProjection(srs.ExportToWkt())
            
            ndv=0#No data value
            ds.GetRasterBand(1).SetNoDataValue(ndv)
            ds = None
            
            if cam=='3' or cam=='4' or cam=='5':
                elem_mosaico[i].append(fn)
            else:
                elem_mosaico.append(fn)

    x_1   = res_georef[9]
    x_2   = res_georef[10]
    x_3   = res_georef[11]
    x_4   = res_georef[12]
    h12   = res_georef[1]
    h32   = res_georef[2]
    x_min = res_georef[13]
    y_min = res_georef[14]
    
    if yawDeg < 0:
        yawDeg = 360 + yawDeg
    yawDeg = 360 - yawDeg
    
    poligono = Polygon([(x_1[0],x_1[1]), (x_3[0],x_3[1]), (x_4[0],x_4[1]),(x_2[0],x_2[1])]) 
    
    for j in range(len(X_points)):
        
        punto = Point(X_points[j], Y_points[j])
                
        if poligono.contains(punto):
            
            
            print('Imagen ' + str(k) + ' contiene el punto ' + name_points[j])
            
            date_value_points[j].append(date_num)
            
            x=X_points[j]
            y=Y_points[j]
            sz_m=size_meter_window_points[j]
            
            xv_1=x-(sz_m/2)*(np.sin(np.deg2rad(yawDeg))+np.sin(np.deg2rad(90-yawDeg)))
            yv_1=y-(sz_m/2)*(np.cos(np.deg2rad(90-yawDeg))-np.sin(np.deg2rad(90-yawDeg)))
            
            xv_2=x-(sz_m/2)*(np.sin(np.deg2rad(yawDeg))-np.sin(np.deg2rad(90-yawDeg)))
            yv_2=y+(sz_m/2)*(np.cos(np.deg2rad(90-yawDeg))+np.sin(np.deg2rad(90-yawDeg)))
            
            xv_3=x+(sz_m/2)*(np.sin(np.deg2rad(yawDeg))-np.sin(np.deg2rad(90-yawDeg)))
            yv_3=y-(sz_m/2)*(np.cos(np.deg2rad(90-yawDeg))+np.sin(np.deg2rad(90-yawDeg)))
            
            xv_4=x+(sz_m/2)*(np.sin(np.deg2rad(yawDeg))+np.sin(np.deg2rad(90-yawDeg)))
            yv_4=y+(sz_m/2)*(np.cos(np.deg2rad(90-yawDeg))-np.sin(np.deg2rad(90-yawDeg)))
            
            x_v=[xv_1,xv_2,xv_4,xv_3]
            y_v=[yv_1,yv_2,yv_4,yv_3]
            
            px_img=[]
            px_img_r=[]
            
            for i in range(len(x_v)):
                
                p_px_r = [np.dot(h12, np.array([x_v[i], y_v[i], 1]))[0] / np.dot(h12, np.array([x_v[i], y_v[i], 1]))[2],
                        np.dot(h12, np.array([x_v[i], y_v[i], 1]))[1] / np.dot(h12, np.array([x_v[i], y_v[i], 1]))[2]]
                
                # p_px_r = [np.dot(h32, np.array([p_px_1[0], p_px_1[1], 1]))[0] / np.dot(h32, np.array([p_px_1[0], p_px_1[1], 1]))[2],
                #         np.dot(h32, np.array([p_px_1[0], p_px_1[1], 1]))[1] / np.dot(h32,     np.array([p_px_1[0], p_px_1[1], 1]))[2]]
                
                p_px_r[0]=p_px_r[0]-x_min
                p_px_r[1]=p_px_r[1]-y_min
                
                p_px = [np.dot(h_final_1, np.array([p_px_r[0], p_px_r[1], 1]))[0] / np.dot(h_final_1, np.array([p_px_r[0], p_px_r[1], 1]))[2],
                        np.dot(h_final_1, np.array([p_px_r[0], p_px_r[1], 1]))[1] / np.dot(h_final_1, np.array([p_px_r[0], p_px_r[1], 1]))[2]]
                
                px_img_r.append(p_px_r)
                px_img.append(p_px)
    
            px_img=np.array(px_img)
            px_img_r=np.array(px_img_r)
            
            for i in range(px_img.shape[0]):
                
                if px_img[i,1]<0:
                    px_img[i,1]=0
                elif px_img[i,1]>height:
                    px_img[i,1]=height
            
                if px_img_r[i,1]<0:
                    px_img_r[i,1]=0
                elif px_img_r[i,1]>height_rec:
                    px_img_r[i,1]=height_rec
                    
                if px_img[i,0]<0:
                    px_img[i,0]=0
                elif px_img[i,0]>width:
                    px_img[i,0]=width
                    
                if px_img_r[i,0]<0:
                    px_img_r[i,0]=0
                elif px_img_r[i,0]>width_rec:
                    px_img_r[i,0]=width_rec
                    
            # sz=size_pixel_window_points[j]
            # ventana=[]
            # ventana.append([p1_px[0]-np.floor(sz/2),p1_px[1]-np.floor(sz/2)])
            # ventana.append([p1_px[0]+np.floor(sz/2),p1_px[1]-np.floor(sz/2)])
            # ventana.append([p1_px[0]+np.floor(sz/2),p1_px[1]+np.floor(sz/2)])
            # ventana.append([p1_px[0]-np.floor(sz/2),p1_px[1]+np.floor(sz/2)])

            # ventana=np.array(ventana)
            
            # for i in range(ventana.shape[0]):
            #     if ventana[i,0]<0:
            #         ventana[i,0]=0
            #     elif ventana[i,0]>width:
            #         ventana[i,0]=width
                
            #     if ventana[i,1]<0:
            #         ventana[i,1]=0
            #     elif ventana[i,1]>height:
            #         ventana[i,1]=height
            
            
            if img_type==1:
                
                for i in range(len(imRec)):              
                    
                    imarray=img[:,:,i]
                    I_value=apply_mask(imarray,px_img)
                    data_value=I_value[I_value>0]
                    mean_value=np.mean(data_value)
                    
                    if cam=='3' or cam=='4' or cam=='5':
                        Value_points[j][i].append(mean_value)   
                    else:
                        Value_points[j].append(mean_value)
                        
            elif img_type==2:
                
                imarray=np.array(I_np2, dtype=np.float32)
                # id=np.where(imarray>0)
                
                # if tipo_pendiente== '1' or tipo_pendiente=='2':
                #     imarray[id]=imarray[id]+pendiente*(k-int(range_im[0]))
                
                I_value=apply_mask(imarray,px_img)
                data_value=I_value[I_value>0]
                mean_value=np.mean(data_value)
                
                if cam=='3' or cam=='4' or cam=='5':
                    Value_points[j][i].append(mean_value)   
                else:
                    Value_points[j].append(mean_value)
                
            elif img_type==3:
                
                for i in range(len(imRec)):              
                    
                    imarray=img[:,:,i]
                    I_value=apply_mask(imarray,px_img)
                    data_value=I_value[I_value>0]
                    mean_value=np.mean(data_value)
                    
                    if cam=='3' or cam=='4' or cam=='5':
                        Value_points[j][i].append(mean_value)
                    else:
                        Value_points[j].append(mean_value)
            
            if cam=='3' or cam=='4' or cam=='5':
                for i in range(len(imRec)):
                    print('La intensidad de la imagen '+str(i)+'en la banda '+str(i)+' es: '+str(Value_points[j][i]))
                    
            else:
                print('La intensidad de la imagen '+str(i)+'en la banda '+str(band)+' es: '+str(Value_points[j][-1]))
                
            if flag_mask:
                
                imarray=np.ones((height_rec,width_rec))
                imarray=apply_mask(imarray,px_img_r)
                                
                [h,b]=imarray.shape
                
                # Create raster for Geotiffs
                fn=DstPathGeo+'/'+'mask_point_'+name_points[j]+"_"+Prefix+str(k)+"_Georef"+".JPG"
                driver = gdal.GetDriverByName('GTiff')
                ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
                ds.GetRasterBand(1).WriteArray(imarray)

                # Apply geotransform
                geot = [x1, resX, x_skew, y1, y_skew, resY]
                ds.SetGeoTransform(geot)
                srs = gdal.osr.SpatialReference()
                
                ###############################
                ##---------------------------##
                ##-- Define UTM projection --##
                ##---------------------------##
                ###############################
                
                # North=1 ó South=0
                
                srs.SetUTM(21,0)
                srs.SetWellKnownGeogCS('WGS84')
                ds.SetProjection(srs.ExportToWkt())
                
                ndv=0#No data value
                ds.GetRasterBand(1).SetNoDataValue(ndv)
                ds = None
    
    # Display bar progress
    bar.update(k-range_im[0]+1)
    sleep(0.1)

for j in range(len(X_points)):
    
    df_point=pd.DataFrame({'Date':date_value_points[j],'intensidad':Value_points[j]})

if img_type==2:
    
    plt.figure()
    plt.plot(temp_mean)
    plt.plot(temp_5,linestyle='dashed')
    plt.plot(temp_95,linestyle='dashed')
    plt.show()

    # Generating a .csv with results of temperature  in the specifics point.
    df=pd.DataFrame({'temp_media':temp_mean,'temp_5':temp_5,'temp_95':temp_95})
    df.to_csv(DstPathGeo+'/''temp.csv',sep=',')

##########################################################
##------------------------------------------------------##
##-- Create mosaic with Geo-transformation parameters --##
##------------------------------------------------------##
##########################################################

if img_type==1 or img_type==3:
    if cam== '1':
        files_to_mosaic = elem_mosaico
        Mos_c=DstPathGeo+'/Mosaic_'+str(band)+".tif"
        warp_options = gdal.WarpOptions(format="GTiff")# ["COMPRESS=LZW"]
        g = gdal.Warp(Mos_c, files_to_mosaic, format="GTiff",
                    options=warp_options,srcNodata=0, dstNodata=0)
        g = None
        
    else:
        for i in range(3):
            files_to_mosaic = elem_mosaico[i]
            warp_options = gdal.WarpOptions(format="GTiff")# ["COMPRESS=LZW"]
            Mos_i=DstPathGeo+'/Mosaic_'+str(i)+".JPG"
            g = gdal.Warp(Mos_i, files_to_mosaic, format="GTiff",
                        options=warp_options,srcNodata=0, dstNodata=0)
            g = None
else:
    
    files_to_mosaic = elem_mosaico
    Mos_c=DstPathGeo+'/Mosaic_'+str(band)+".tif"
    warp_options = gdal.WarpOptions(format="GTiff")# ["COMPRESS=LZW"]
    g = gdal.Warp(Mos_c, files_to_mosaic, format="GTiff",
                options=warp_options,srcNodata=0, dstNodata=0)
    g = None

OutputPathCSV_1=DstPathGeo+'/date_value_points.csv'

with open(OutputPathCSV_1, 'w', newline='') as f:
    writer = csv.writer(f)
    
    for name, row in zip(name_points, date_value_points):
        writer.writerow([name] + row)  # [S1, value]

OutputPathCSV_2=DstPathGeo+'/Value_points.csv'

with open(OutputPathCSV_2, 'w', newline='') as f:
    writer = csv.writer(f)
    
    for name, row in zip(name_points, Value_points):
        new_row = [name]  # first column is name
        for item in row:
            if isinstance(item, list):
                new_row.append(",".join(map(str, item)))  # sublists -> "v1,v2,v3"
            else:
                new_row.append(str(item))  # number
        writer.writerow(new_row)

# if __name__ == "__main__":
#     main()