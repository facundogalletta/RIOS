"""
Author: Facundo Galletta, fgalltta@fing.edu.uy
Date: 3/20/2024
Description: This code performs georectification and georeferencing of images taken from a drone, utilizing other auxiliary functions included with the code.
"""
from __future__ import print_function, unicode_literals
import os
os.environ['PROJ_LIB'] = r'C:/Users/fgalletta/anaconda3/envs/geomapper/Library/share/proj'
from PIL import Image
import cv2 as cv
import numpy as np
from osgeo import gdal
import shutil
import matplotlib.pyplot as plt
import pandas as pd
import csv

from core.interseccion_fotos import intersection
from core.georectification_general import georef_img
from core.interactive_windows import interactive_window, interactive_window_number,seleccionar_carpeta,seleccionar_csv,interactive_window_aviso,interactive_window_numbers
from core.micasense import capture as capture
from core.background import background
from core.prefijo import prefijo
from core.aplicar_mascara import aplicar_mascara
import glob
from shapely.geometry import Polygon, Point

# Funciones para la barra de display
import progressbar
from time import sleep


"""
Esta es la version no interactiva (en gran parte) del codigo. Se puede acceder a la versión interactiva en Geo MapperDron. 

Se debe acceder la información de:

- SourcePath: Ruta de las imagenes georeferenciadas
- SkyNames: Ruta de las imagenes del cielo
- PanelNames: Ruta de las imagenes del panel - Usualmente es None porque se usan capturas de la Micasense que incluyen DLS

- DstPathGeo: Ruta de las imagenes georeferenciadas
- DstPathRect: Ruta de las imagenes rectificadas
- csv_file: Ruta del csv con las coordenadas de las imagenes georeferenciadas - Ingresar si se estan u

- type_process: Tipo de procesamiento
- cam: Cámara utilizada
- save: Guardar las imagenes rectificadas
- [ini,fin]: Rango de imagenes a procesar
- corregir: Insdicar si se corrie la deriva de las imagenes termicas
- tipo: Tipo de corrección de efectos de borde.
- [ini_c,fin_c]: Rango de imagenes de la termica en agua fría. 
- flag_plot_promedio: Graficar promedio de agua fría utilizado para la corrección de borde.
- temp_i: Temperatura media de la primera imagen.
- tipo_pendiente: Tipo de pendiente de corrección de los efectos de borde en la intersección de imagenes.

"""

"""
Incluye el cambio de estructura iniciando con la definición del tipo de procesamiento que desee hacer:

- Op 1 : Georeferenciar imagenes RGB de cualquier DRON.
- Op 2 : Georeferenciar imagenes de una cámara infrarojo-térmica.
- Op 3 : Georreferenciar imagenes de diferfentes longitudes de onda para calcular reflectancia.

"""

# SOURCE PATH
SourcePath='D:/GeoMapperDron/PruebaTermica/1/'
csv_file = 'D:/GeoMapperDron/PruebaTermica/1/flight_record_FLY219.csv'

SkyNames = ['D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_1.tif',
            'D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_2.tif',
            'D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_3.tif',
            'D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_4.tif',
            'D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_5.tif',
            'D:/GeoMapperDron/PruebaPuntadelTigre/20210520/Altum/Cielo/IMG_0037_6.tif'
            ]

PanelNames = None

# Definir tipo de proceso

type_process='2'                # ['1 - Georreferenciar Imagens RGB','2 - Georreferenciar Imagens Termicas', 2 - Georreferenciar Imagens Multiespectrales y aplicar Radiometria']
cam='2'                         # ['1 - Micasense Altum (multiespectral)','2 - Micasense Altum (infrarrojo térmico)','3 - Zenmuse X5S','4 - DJI Mavic Pro','5 - DJI Mini 2','6 - Definir parámetros manualmente']
band=6                          # ['1 - Blue','2 - Green','3 - Red','4 - RedEdge','5 - NIR']                        
save='2'                        # Imagenes rectificadas ['1 - Si','2 - No']
[ini,fin]=[55,225]

# Puntos de donde se quieren obtener datos
X_points=[542676.833,542685.372,542717.086,542725.625,542546.06,542594.55,542807.905,542856.395]
Y_points=[6153551.122,6153517.179,6153391.107,6153357.164,6153415.10,6153427.29,6153480.979,6153493.177]
size_pixel_window_points=[3,3,3,3,3,3,3,3]
size_meter_window_points=[3,3,3,3,3,3,3,3]
name_points=['S1','S2','S3','S4','S5','S6','S7','S8']
flag_mask=False

# Cosas que se deben modificar solo en caso de usar la banda termica.
corregir='1'                    # Deriva ['1 - Si','2 - No']
tipo='1'                        # Efectos de borde ['1 - Original','2 - Paraboloide','3 - Gaussiana','4 - Ninguna']
[ini_c,fin_c]=[20,55]
range_c=[int(ini_c),int(fin_c)] # Rango de agua fría
flag_plot_promedio='2'          # Graficar promedio de agua fría ['1 - Si','2 - No']
temp_i = 22.2                   # Temperatura media de la primera imagen
tipo_pendiente='3'              # Tipo de pendiente ['1 - Actual, 2 - Ingresar','3 - Ninguna']
pendiente_calibrada=-0.013#(22.36-22.18)/(65-238)+0.0001 -0.0247
fecha_pendiente_calibrada = '16/12/2020 - 14:00'
DT_promedio_agua_fria=[]


if type_process=='3':
    
    DstPathGeo = SourcePath+'/Geo_Ref_'+str(band)+'_'+str(ini)+'_'+str(fin)
    DstPathRect = SourcePath+'/Rec_Ref_' +str(band)+'_'+str(ini)+'_'+str(fin)

elif type_process=='2':
    
    DstPathGeo = SourcePath+'/Geo_Term'
    DstPathRect = SourcePath+'/Rec_Term'
    
    if corregir=='1':
        
        DstPathGeo=DstPathGeo+'_D'
        DstPathRect=DstPathRect+'_D'
              
        if tipo=='1':
            DstPathGeo=DstPathGeo+'_A'
            DstPathRect=DstPathRect+'_A'
        elif tipo =='2':
            DstPathGeo=DstPathGeo+'_P'
            DstPathRect=DstPathRect+'_P'
        elif tipo=='3':
            DstPathGeo=DstPathGeo+'_G'
            DstPathRect=DstPathRect+'_G'
        elif tipo=='4':
            DstPathGeo=DstPathGeo+'_NE'
            DstPathRect=DstPathRect+'_NE'
        
        if tipo_pendiente=='1' or tipo_pendiente=='2':
            DstPathGeo=DstPathGeo+'_S'
            DstPathRect=DstPathRect+'_S'
        else:
            DstPathGeo=DstPathGeo+'_NS'
            DstPathRect=DstPathRect+'_NS'
            
    else:
        DstPathGeo=DstPathGeo+'_RAW'
        DstPathRect=DstPathRect+'_RAW'
    
    DstPathGeo=DstPathGeo+'_'+str(ini)+'_'+str(fin)
    DstPathRect=DstPathRect+'_'+str(ini)+'_'+str(fin)
        
elif type_process=='1':
    
    DstPathGeo = SourcePath+'/Geo_'+str(band)+'_'+str(ini)+'_'+str(fin)
    DstPathRect = SourcePath+'/Rec_'+str(band)+'_'+str(ini)+'_'+str(fin)
    
if type_process=='1': # Proceso de georreferenciar una imagen RGB o una banda de una imagen multiespectral
    
    if cam=='1': # Camara Micasense Altum
        
        width=2015#2064 #(px)
        height=1496#1544 #(px)
        FOVwidth=48 #(°)
        FOVheight=37 #(°)
        prefix="IMG_"
        img_type=1
    
    if cam=='2': # Camara Micasense Altum (infrarrojo termico)
    # Informar que el sensor termico no es type_process='1' y terminara el codigo
        print('\n¡No seleccionó una opcion valida! El sensor termico no le corresponde este procesamiento.\n')
    
    if cam=='3': # Camara RGB
        
        width=5280
        height=2970
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        prefix="DJI_"
        band=0
        img_type=1

    elif cam == '4': # Camara RGB
        
        width=4000
        height=2250
        FOVwidth=71.19928287983431
        FOVheight=43.86979719641275
        prefix="DJI_"
        band=0
        img_type=1

    elif cam == '5': # Camara RGB
        width=4000
        height=2250
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        prefix="DJI_"
        band=0
        img_type=1
    
    elif cam=='6': # Una nueva camara
        
        parametro='Width (px)'
        width=int(interactive_window_number(parametro))
        
        parametro='Height (px)'
        height=int(interactive_window_number(parametro))
        
        parametro='Field of view in the long side (width)'
        FOVwidth=int(interactive_window_number(parametro))
        
        parametro='Field of view in the short side (height)'
        FOVheight=int(interactive_window_number(parametro))
        prefix=interactive_window(['Ejemplos','DJI_','IMG_'], 'Definir el prefijo de las imagenes antes de la numeración')
        img_type=1
        
elif type_process=='2': # Georreferenciar una imagen termica
    
    if cam=='2': #Camara Micasense Altum (infrarrojo termico)
        width=160
        height=120
        FOVwidth=57
        FOVheight=44
        prefix="IMG_"
        band=6
        img_type=2
    else:
        print('\n¡No seleccionó una opcion valida!\n')
        
elif type_process=='3': # Proceso de georreferenciar una imagen en reflectancia desde una camara RGB o una banda de una imagen multiespectral
    print('entra')
    if cam=='1': # Camara Micasense Altum
        width=2015#2064 #(px)
        height=1496#1544 #(px)
        FOVwidth=48 #(°)
        FOVheight=37 #(°)
        prefix="IMG_"
        img_type=3
        print('entra')
    
    elif cam=='2': # Camara Micasense Altum (infrarrojo termico)
        print('\n¡No seleccionó una opcion valida! El sensor termico no le corresponde este procesamiento.\n')
    
    elif cam=='3': # Camara RGB
        width=5280
        height=2970
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        prefix="DJI_"
        band=0
        img_type=3

    elif cam == '4': # Camara RGB
        width=4000
        height=2250
        FOVwidth=71.19928287983431
        FOVheight=43.86979719641275
        prefix="DJI_"
        band=0
        img_type=3

    elif cam == '5': # Camara RGB
        width=4000
        height=2250
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        prefix="DJI_"
        band=0
        img_type=3

    elif cam=='6': # Una nueva camara
        
        parametro='Width (px)'
        width=int(interactive_window_number(parametro))
        
        parametro='Height (px)'
        height=int(interactive_window_number(parametro))
        
        parametro='Field of view in the long side (width)'
        FOVwidth=int(interactive_window_number(parametro))
        
        parametro='Field of view in the short side (height)'
        FOVheight=int(interactive_window_number(parametro))
        prefix=interactive_window(['Ejemplos','DJI_','IMG_'], 'Definir el prefijo de las imagenes antes de la numeración')
        img_type=1
            
    else:
        print('\n¡No seleccionó una opción válida!\n')
        pass

################################
# Directorios de procesamiento #
################################

if cam=='3' or cam=='4' or cam=='5':
    SourceImags=glob.glob(os.path.join(SourcePath, '*.JPG'))    
elif cam=='1' or cam=='2':
    SourceImags=glob.glob(os.path.join(SourcePath, '*.tif'))

if save == '1':
    # Si el path existe, limpiar; si no, crearlo
    if os.path.exists(DstPathRect):
        if os.listdir(DstPathRect):  # si no está vacío
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
    
# Si el path existe, limpiar todo
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
    # Si no existe, crearlo
    print(f"Creating directory: {DstPathGeo}")
    os.makedirs(DstPathGeo, exist_ok=True)

if cam=='3' or cam=='4' or cam=='5':
    csv_file=[]

NameTXT='Data.txt' 
txtAux='Aux.txt'

#########################
# RANGO DE PROCEAMIENTO #
#########################

range_im=[int(ini),int(fin)]
OutputPathTXT= DstPathGeo+'/'+NameTXT

#################
# Procesamiento #
#################

# Parámetros de procesamiento para el caso que sean imagenes de la banda térmica

if img_type==2:
      
    temp_mean=[]
    temp_5=[]
    temp_95=[]
    coef_corr=[]
    
    if corregir == '1':

        if tipo=='1':
            correction_bordes='Original'

        elif tipo=='2':
            correction_bordes='Paraboloide'

        elif tipo=='3':
            correction_bordes='Gaussian'

        elif tipo=='4':
            correction_bordes=None
        else:
            interactive_window_aviso('''NO SELECCIONO UNA OPCION VALIDA''' )
            pass
            
        if correction_bordes=='Paraboloide' or correction_bordes=='Gaussian' or correction_bordes=='Original':
            Prefix='IMG_'
            path=[]
            for k in range(range_c[0],range_c[-1]+1):
                name=os.path.join(SourcePath, prefijo(k,Prefix)+str(k)+"_"+str(band)+".tif")
                path.append(name)
            promedio_agua_fria=background(path,band)
            DT_promedio_agua_fria=promedio_agua_fria-np.min(promedio_agua_fria)
            
            if flag_plot_promedio=='1':
                plt.figure()
                plt.imshow(promedio_agua_fria,cmap='jet')
                plt.colorbar()
                plt.show()
            
        if tipo_pendiente=='1':
            pendiente=pendiente_calibrada
        elif tipo_pendiente=='2':
            pendiente=interactive_window_number('Enter the new slope')
        elif tipo_pendiente=='3':
            pendiente=0
            
    else:
        
        pendiente=0
        correction_bordes=None
        DT_promedio_agua_fria=[]
        tipo_pendiente='3'
        
else:
    corregir='2'
    correction_bordes=None
    DT_promedio_agua_fria=[]
    pendiente=0
        
# .TXT FROM GDAL       

g=open(DstPathGeo+'/'+txtAux,"w+")
g.write("""RangoFotosi RangoFotosf Band type
"""+str(range_im[0])+' '+str(range_im[-1])+' '+str(band)+ ' ' + str(img_type)+'\n')
g.close()

if cam=='3' or cam=='4' or cam=='5':
    prefix="DJI_"
else:
    prefix="IMG_"
         
#Initialize mosaic

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
    
    #Define the patho of the image and the capture of image
    
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
        
    #Load imagel PIL and binary
    pil_img = Image.open(name)
    binary_img = open(name,"rb")
    
    # RECTIFICACIÓN DE IMAGEN
    res_georef=georef_img(img_type,cam,height, width, FOVwidth, FOVheight,capture_img, pil_img, binary_img,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes,PanelNames,SkyNames)
    img=res_georef[0]
    imRec=res_georef[3]
    height_rec=res_georef[4]
    width_rec=res_georef[5]
    resHeight=res_georef[6]
    resWidth=res_georef[7]
    pt_sup_izq=res_georef[8]
    yawDeg=res_georef[15]
    date_num=res_georef[16]
    h_final=res_georef[17]
    h_final_1=np.linalg.inv(h_final)
    
    
    # Correcciiones de la banda termica
    if corregir == '1':
        if not(range_im[0]==range_im[1]):
            if img_type==2:    
                if k == range_im[0]:
                    img=np.array(img)
                    size_ventana=(30,30) # Dimenciones de una ventana centrada en la imagen
                    m2=np.mean(img[int(img.shape[0]/2-size_ventana[0]/2):int(img.shape[0]/2+size_ventana[0]/2),int(img.shape[1]/2-size_ventana[1]/2):int(img.shape[1]/2+size_ventana[1]/2)]) # calcular la media en una ventana centrada en el centro de la imagen
                    coef_corr.append(temp_i/m2)
                    im_res = cv.multiply(np.array(imRec[0]).astype(np.float32),np.float64(coef_corr[-1]))
                    I_np2=cv.multiply(np.array(img),np.float64(coef_corr[-1]))
                else:
                    name_0=os.path.join(SourcePath, prefijo(k-1,prefix)+str(k-1)+"_"+str(band)+".tif")
                    px_1,px_2, px_1r, px_2r,px_bar1,px_bar2,im2Rec,im1,im2 = intersection(img_type,cam,name_0,name, FOVwidth,FOVheight,band,csv_file,flag_features='d',
                                                                                                                flag_interseccion=False,DT_promedio_agua_fria=DT_promedio_agua_fria,correction_bordes=correction_bordes,panel_names=PanelNames,sky_names=SkyNames)
                    I_np1 = cv.multiply(np.array(im1),np.float64(coef_corr[-1]))
                    I_np2=np.array(im2)
                    
                    Ir_1 = aplicar_mascara(I_np1,px_1)
                    Ir_2 = aplicar_mascara(I_np2,px_2)
                    
                    data_1=Ir_1[Ir_1>0]
                    data_2=Ir_2[Ir_2>0]
                    
                    data_filtered_1=cv.GaussianBlur(data_1, (5, 5), 0)
                    data_filtered_2=cv.GaussianBlur(data_2, (5, 5), 0)
                    
                    #Calculo estadisticos de la region en ambas imagenes 
                    
                    m1=np.mean(data_filtered_1)
                    m2=np.mean(data_filtered_2)
                    
                    coef_corr.append(m1/m2)
                    
                    I_np2=cv.multiply(np.array(im2),np.float64(coef_corr[-1]))
                    im_res = cv.multiply(np.array(im2Rec[0]).astype(np.float32),np.float64(coef_corr[-1]))

                    # LLevo a cero valores de bordes interpolados en rectificación "corregidos". Evito bordes en mosaico.
                    im_res[np.where(im_res<temp_i-10)]=0
        else: 
            if img_type==2:
                im_res=imRec[0]
    else:
        I_np2=np.array(img)
        im_res=imRec[0]        
       
    x_1=float(pt_sup_izq[0])
    y_1=float(pt_sup_izq[1])
    
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
    

    # Parámetros de geotransformación
       
    x1=d[1]
    y1=d[2]
    resX=d[3]
    resY=d[4]
    yawDeg=d[5]
    angle=yawDeg*np.pi/180
    x_skew=-resY*np.sin(angle)
    y_skew=-resX*np.sin(angle)
    resX=resX*np.cos(angle)
    resY=-resY*np.cos(angle)
    
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
        
        if tipo_pendiente== '1' or tipo_pendiente=='2':
            imarray[id]=imarray[id]+pendiente*(k-int(range_im[0]))
            I_np2[id_2]=I_np2[id_2]+pendiente*(k-int(range_im[0]))
        
        #Dimensiones de la imagen
        [h,b]=imarray.shape 
        
        #Creación del ráster que contendrá el Geotiff:
        fn=DstPathGeo+'/'+Prefix+str(k)+"_"+str(band)+"_georeferenced"+".tif"
        driver = gdal.GetDriverByName('GTiff')
        ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(imarray)

        #Definición de parámetros de georreferenciación:
        geot = [x1, resX, x_skew, y1, y_skew, resY]#[500000, 10, 0, 4600000, 0, -10]
        ds.SetGeoTransform(geot)
        srs = gdal.osr.SpatialReference()
        
        # Definir sistema de coordenadas. El segundo parámetro es North=1 ó South=0
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
        
        print('Temperatura media en la imagen ' + str(k) + ' es de ' + str(temp_mean[-1]))
        
    elif img_type==3:
        
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

    x_1=res_georef[9]
    x_2=res_georef[10]
    x_3=res_georef[11]
    x_4=res_georef[12]
    h12=res_georef[1]
    h32=res_georef[2]
    x_min=res_georef[13]
    y_min=res_georef[14]
    
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
                    I_value=aplicar_mascara(imarray,px_img)
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
                
                I_value=aplicar_mascara(imarray,px_img)
                data_value=I_value[I_value>0]
                mean_value=np.mean(data_value)
                
                if cam=='3' or cam=='4' or cam=='5':
                    Value_points[j][i].append(mean_value)   
                else:
                    Value_points[j].append(mean_value)
                
            elif img_type==3:
                
                for i in range(len(imRec)):              
                    
                    imarray=img[:,:,i]
                    I_value=aplicar_mascara(imarray,px_img)
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
                imarray=aplicar_mascara(imarray,px_img_r)
                                   
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
                
                # Set projection. The second parameter is: North=1 or South=0
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

    # Generar .csv con resultados de temp
    df=pd.DataFrame({'temp_media':temp_mean,'temp_5':temp_5,'temp_95':temp_95})
    df.to_csv(DstPathGeo+'/''temp.csv',sep=',')
    

"""------------ Armado de mosaico ------------"""
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



# Ejemplo: date_value_points = [[1.2], [3.4], [5.6]]


OutputPathCSV_1=DstPathGeo+'/date_value_points.csv'

with open(OutputPathCSV_1, 'w', newline='') as f:
    writer = csv.writer(f)
    
    for name, row in zip(name_points, date_value_points):
        writer.writerow([name] + row)  # [S1, valor]

OutputPathCSV_2=DstPathGeo+'/Value_points.csv'

with open(OutputPathCSV_2, 'w', newline='') as f:
    writer = csv.writer(f)
    
    for name, row in zip(name_points, Value_points):
        new_row = [name]  # primer columna = nombre del punto
        for item in row:
            if isinstance(item, list):
                new_row.append(",".join(map(str, item)))  # sublistas -> "v1,v2,v3"
            else:
                new_row.append(str(item))  # número
        writer.writerow(new_row)