""" Script para georreferenciación y creación de GeoTifs de cada imagen. Y creado del mosaico de las imagenes.

     Inputs:
        Archivo .tif de cada imagen rectificada corregida.
        Archivo .txt auxiliar con información de la correccion.
        Archivo .txt con coordenadas UTM de los vértices sup. izq. y con la resolución espacial de los píxeles en horizontal y vertical.
        
     Outputs:
        Geotif de cada imagen rectificada corregida georeferenciada.
        Mosaico de estos Geotif's.
"""
import os
os.environ['PROJ_LIB'] = r'C:/Users/fgalletta/anaconda3/envs/timag/Library/share/proj'
from osgeo import gdal
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage.io import imread
from interactive_windows import interactive_window, interactive_window_number,seleccionar_carpeta,seleccionar_imagen
import PySimpleGUI as sg
import cv2


# INGRESAR UBICACION DEL P´ROS PARA USA OSGEO, SE DEBE CAMBIAR EN CADA COMPUTADORA. DE LO CONTRARIO, NO GENERA LOS GEOTIF'S NI EL MOSAICO.


#############################################################################################
# Definir ubicacion de imagenes rectificadas corregidas, de .txt auxiliares y de resultados.#
#############################################################################################

lista = ['Seleccionar rutas','Usar rutas de codigo']
parametro = 'Rutas usadas'

opcion=interactive_window(lista,parametro)

if opcion == 'Seleccionar rutas':
    # Path a las imágenes a georeferenciar.
    
    parametro='la ruta de las imagenes a georeferenciar.'
    
    path_from_image=seleccionar_carpeta(parametro)
    path_from_image=path_from_image+'/'

    # Path al archivo txt con la información necesaria para georreferenciar las fotos.
    parametro='la ruta del archivo txt con la información necesaria para georreferenciar las fotos.'

    path_from_txt=seleccionar_carpeta(parametro)
    path_from_txt=path_from_txt+'/'
    
    # Path para outputs (que serán geotiffs).
    # Son imagenes que se pueden cargar como un raster. Como por ejempo, en QGis.
    parametro='la ruta para outputs (que serán geotiffs).'

    path_to_GTIF = seleccionar_carpeta(parametro)
    path_to_GTIF=path_to_GTIF+'/'
    
    parametro= 'Nombre del mosaico a guardar'
    nombre=interactive_window_number(parametro)
    # Nombre del mosaico que se creará:
    Mos=path_to_GTIF+nombre
else:
    path_from_image='../Descargando/Rectificadas/P1/'
    path_from_txt='../Descargando/Rectificadas/P1/'
    path_to_GTIF='../Descargando/Georeferenciadas/P1/Equalizada/'
    Mos='p202310271754_DRON'

# Cargar archivo que contiene datos de que imagenes fueron corregidas

d=np.loadtxt(os.path.join(path_from_txt,"Auxiliar.txt"), skiprows=1,encoding='latin1')

rangoFotosi=int(d[0])
rangoFotosf=int(d[1])
img_type=int(d[-1])

if img_type==1 or img_type==2:
    correction_bordes=int(d[3])
    band=int(d[2])
else:
    correction_bordes=None
    
# band=int(d[2])
# correction_bordes=int(d[3])

#Cargar archivo que contiene datos para georeferenciación de cada imagen y creado de mosaico. 

d=np.loadtxt(path_from_txt+"Datos.txt", skiprows=0,encoding='latin1') #Carga archivo txt
##############################################
#Funcion para obtener el prefijo de las fotos#
##############################################

if img_type==3:
    prefix="DJI_"
else:
    prefix="IMG_"

def prefijo(k):
    """ Funcion para obtener el prefijo de las fotos tomadas con la camara micasense.

    Args:
        k: Numero de foto
    Returns:
        string: Prefijo de la foto.
    """
     
    if k<10:
        Prefix=prefix+'000'
    elif k<100:
        Prefix=prefix+'00'
    elif k<1000:
        Prefix=prefix+'0'
    else:
        Prefix=prefix

    return Prefix

###################################################
# Definir coomo se realizo la correccion de bordes#
###################################################

#Inicio lista que tendrá nombres de archivos a unir en el mosaico

elem_mosaico=[]

##########################################################################################
# Iterción para georeferenciar cada imagen y agregarla a la lista de nombres del mosaico #

Mos_i=path_to_GTIF+Mos+".JPG"

for k in range(rangoFotosi,rangoFotosf+1):
    
    #Cargar la imagen a georeferenciar
    Prefix=prefijo(k)
    rgb=[]
    for i in range(3):
        im=imread(path_from_image+"orthorectified_"+Prefix+str(k)+'_'+str(i)+".JPG")
        rgb.append(np.array(im))
    rgb=np.array(rgb)
    
    eq = np.array(rgb[0]-rgb[2])
    
    #Dimensiones de la imagen
    [h,b]=eq.shape 

    # Cargar coordenada x,y UTM del vertice superior izquierdo de la imagen rectificada corregida.
    x1=d[k - rangoFotosi][1]
    y1=d[k - rangoFotosi][2]
    
    # Cargar la resolución en la vertical y horizontal de cada imagen rectificada corregida.
    resX=d[k - rangoFotosi][3]
    resY=d[k - rangoFotosi][4]
    
    # Cargar el angulo de orientacion (de la vetical) de cada imagen rectificada corregida.
    yaw=d[k - rangoFotosi][5]
    angle=yaw*np.pi/180

    #Cálculos para rotación de la imagen en el geotif:
    
    x_skew=-resY*np.sin(angle)
    y_skew=-resX*np.sin(angle)
    resX=resX*np.cos(angle)
    resY=-resY*np.cos(angle)

    #Creación del ráster que contendrá el Geotiff:
    fn=path_to_GTIF+Prefix+str(k)+"_"+str(i)+"_georeferenced_"+".JPG"
        
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(fn, xsize=b, ysize=h, bands=1, eType=gdal.GDT_Float32)
    ds.GetRasterBand(1).WriteArray(eq)

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

    elem_mosaico=elem_mosaico+[fn]
    
"""------------ Armado de mosaico ------------"""
files_to_mosaic = elem_mosaico
g = gdal.Warp(Mos_i, files_to_mosaic, format="GTiff",
            options=["COMPRESS=LZW"],srcNodata=0, dstNodata=0)
g = None
                