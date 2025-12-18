from __future__ import print_function, unicode_literals
import os
from PIL import Image
import cv2 as cv

from ArmadoMosAuxiliares import orientacion, str2lat_or_long_vec, LongLat2UTM, str2fraction, datetimeSTR2timeFLOAT, fracc_sec
from correction_distorted_img import calibration
from calibration_dji import calibration_dji
import micasense.capture as capture
from PIL.ExifTags import TAGS
from georectification_general import generacion_nueva_img_general


while True:
    cam=input('''Ingrese cámara usada:
        1-Micasense Altum (multispectral)
        2-Micasense Altum (thermal)
        3-DJI Inspire 2
        4-Ingresar parámetros de cámara manualmente
        Opción:''')

    if cam=='1':
        width=2015#2064 #(px)
        height=1496#1544 #(px)
        FOVwidth=48 #(°)
        FOVheight=37 #(°)
        band=input('''Ingrese número de banda:
             1-Blue
             2-Green
             3-Red
             4-RedEdge
             5-NIR
             Opción:''')
        img_type=1
        break
    elif cam=='2':
        width=2015
        height=1469
        FOVwidth=57
        FOVheight=44
        band=6
        img_type=2
        break
    elif cam=='3':
        width=5280
        height=2970
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        band=0
        img_type=3
        break
    elif cam=='4':
        width=int(input('Width (px):'))
        height=int(input('Height (px):'))
        FOVwidth=int(input('FOV en el lado largo (°):'))
        FOVheight=int(input('FOV en el lado corto (°):'))
        break
    else:
        print('''

        No seleccionó una opción válida!!!

        ''')
        pass


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

#################################
# Definir path de procesamiento #
#################################

SourcePath='D:/fgalletta/Inspire2/Inspire 2_ANP/Descargando/'

id_carpeta_save='Prueba2/'

os.mkdir(SourcePath+id_carpeta_save)
DstPath=SourcePath+id_carpeta_save
csv_file = []

NameTXT='Datos.txt' 
txtAux='Auxiliar.txt'

range_im=[412,412]

OutputPathTXT= DstPath+NameTXT

#################
# Procesamiento #
#################

#Crear txt de datos para creado de geotifs
        
if img_type==3:
    g=open(DstPath+txtAux,"w+")
    g.write("""RangoFotosi RangoFotosf Band type
    """+str(range_im[0])+' '+str(range_im[-1])+' '+str(band)+ ' ' + str(img_type)+'\n')
    g.close()
else:
    g=open(DstPath+txtAux,"w+")
    g.write("""RangoFotosi RangoFotosf Band type
    """+str(range_im[0])+' '+str(range_im[-1])+' '+str(band)+' '+str(img_type)+'\n')
    g.close()

# Iteración de for

for k in range(range_im[0],range_im[1]+1):
    Prefix=prefijo(k)
    
    #Segun el tipo de imagen defino el path y lña "captura" de la imágen
    if img_type==3:
        name=os.path.join(SourcePath, Prefix+str(k)+".JPG")
        capture_img=[]
        flag_features='dji'
    else:
        name=os.path.join(SourcePath, Prefix+str(k)+"_"+str(band)+".tif")
        capture_img = capture.Capture.from_filelist([name])
        flag_features='d'
        
    #Cargar imagen PIL y binaria
    
    pil_img = Image.open(name)
    binary_img = open(name,"rb")
    
    #Rectificar la imágen
    img,h12,h32,imReg,h, w,resHeight,resWidth, pt_sup_izq,x1,x2,x3,x4,x_min,y_min,yawDeg=generacion_nueva_img_general(img_type,height, width, FOVwidth, FOVheight,capture_img, pil_img, binary_img,band, csv_file,flag_features,[],None)
   
    x_1=float(pt_sup_izq[0])
    y_1=float(pt_sup_izq[1])
    
    print(x_1,y_1)
    
    if k==range_im[0]:
        f=open(OutputPathTXT,"w+")
        f.write(str(k)+' '+str(x_1)+' '+str(y_1)+' '+str(resWidth)+' '+str(resHeight)+' '+str(yawDeg)+ '\n')
        f.close()
    else:
        f=open(OutputPathTXT,"a+")
        f.write(str(k)+' '+str(x_1)+' '+str(y_1)+' '+str(resWidth)+' '+str(resHeight)+' '+str(yawDeg)+ '\n')
        f.close()
    
    for i in range(len(imReg)): 
        if img_type==3:
            cv.imwrite(os.path.join(DstPath, "orthorectified_"+Prefix+str(k)+"_"+str(i)+".JPG"),imReg[i])  
        else:
            cv.imwrite(os.path.join(DstPath, "orthorectified_"+Prefix+str(k)+"_"+str(band)+".tif"),imReg[i])    
        