from __future__ import print_function, unicode_literals
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from core.correction_distorted_img import calibration
import core.micasense.capture as capture
from core.georectification_general import generacion_nueva_img_general
from core.interactive_windows import interactive_window, interactive_window_number,seleccionar_carpeta,seleccionar_csv,interactive_window_aviso,interactive_window_numbers


prefix="IMG_"
def prefijo(k):    
    if k<10:
        Prefix=prefix+'000'
    elif k<100:
        Prefix=prefix+'00'
    elif k<1000:
        Prefix=prefix+'0'
    else:
        Prefix=prefix

    return Prefix

SourcePath_1='D:/fgalletta/timag/Source Path/prueba/res_int/'
SourcePath='D:/fgalletta/timag 2023/Source Path/prueba/res_int/'

DstPath=SourcePath+"/nueva_prueba/"
csv_file = 'D:/fgalletta/timag 2023/Source Path/prueba/prueba_toldo.csv'

path_imgs=glob.glob(os.path.join(SourcePath, '*.tif'))
path_imgs=sorted(path_imgs, key=lambda x: x.split('/')[-1])


def background(path_imgs):
    width, height=Image.open(path_imgs[0]).size
    img_sum=np.zeros((height,width))
    for path in path_imgs:
        capture_img1 = capture.Capture.from_filelist([path])
        img=calibration(capture_img1, band,height,width,DT_promedio_agua_fria=[],correction_bordes=None)
        img_sum+=img#.astype('float32')
    img=img_sum/len(path_imgs)
    return img


width=5280
height=2970
FOVwidth=64.686982896768800
FOVheight=39.211354832645280
band=0
img_type=3


width=2015#2064 #(px)
height=1496#1544 #(px)
FOVwidth=48 #(°)
FOVheight=37 #(°)
band=1
img_type=1

range_agua_fria=[] 

correction_bordes=None

if correction_bordes is not None:

    while True:
        
        tipo=input('''

                    #############################################
                    # DEFINIR EL RANGO DE IMAGENES EN AGUA FRIA #
                    #############################################

            1- Definir un unico rango de imagenes en agua fria.
            2- Definir mas de un rango de imagenes en agua fria. 

            Opcion:''')

        if tipo=='1':
            ini=input('''
                El numero de imagen inicial es: ''')
            ini=int(ini)
            
            fin=input('''
                El numero de imagen final es: ''')
            
            fin=int(fin)
            
            range_agua_fria.append([ini,fin])
            break
            
        elif tipo=='2':
            cr=input('''
                La cantidad de rangos a usar es: ''')
            cr=int(cr)
            for i in range(cr):
                print('''En el rango ''' +str(i+1))
                ini=input('''
                    El numero de imagen inicial es: ''')
                ini=int(ini)
                fin=input('''
                    El numero de imagen final es: ''')
                fin=int(fin)
                range_agua_fria.append([ini,fin])
            break
        else:
            print('''
            
            NO SELECCIONO UNA OPCION VALIDA
            
            ''')
            pass

    sum_promedios=np.zeros((height,width))

    for i in range(np.size(range_agua_fria,0)):
        sum_promedios+=background(path_imgs[range_agua_fria[i][0]:range_agua_fria[i][1]])

    promedio_agua_fria=sum_promedios/np.size(range_agua_fria,0)
    DT_promedio_agua_fria=promedio_agua_fria-np.min(promedio_agua_fria)
else:
    DT_promedio_agua_fria=[]



while True:
    lista=['Micasense Altum (multispectral)','Micasense Altum (thermal)','DJI Inspire 2','DJI Mavic Pro','Set parameters manually']
    parametro='Select camera used'
    cam=interactive_window(lista, parametro)
    
    if cam=='Micasense Altum (multispectral)':
        width=2015#2064 #(px)
        height=1496#1544 #(px)
        FOVwidth=48 #(°)
        FOVheight=37 #(°)
        
        lista=['Blue','Green','Red','RedEdge','NIR']    
        parametro='Número de banda'
        band=interactive_window(lista, parametro)
        if band=='Blue':
            band=1
        elif band=='Green':
            band=2
        elif band=='Red':
            band=3
        elif band=='RedEdge':
            band=4
        elif band=='NIR':
            band=5
        img_type=1
        break
    
    elif cam=='Micasense Altum (thermal)':
        width=160
        height=120
        FOVwidth=57
        FOVheight=44
        band=6
        img_type=2
        coef_corr=[]
        
        #Aviso que puede haber deriva entre fotos
        interactive_window_aviso(''''¡It can drift in the measurements of temperature with infrared sensor for heating effects.!
                
                En este codigo se implementaron una serie de correcciones descriptas en: Galletta at. all. TELEDETECCIÓN EN CUERPOS DE AGUAS MEDIANTE DRON Y CÁMARA MULTIESPECTRAL. (https://drive.google.com/file/d/1MpSw6opJHa3rp8caTwheuzyqXa012p-i/view)
                ''')
                
                
        lista=['Original','Paraboloide','Gaussian','No']
        parametro='Define the type of correction edge effects to be applied'
        
        tipo=interactive_window(lista, parametro)
        
        if tipo=='Original':
            correction_bordes='Original'
        
        elif tipo=='Paraboloide':
            correction_bordes='Paraboloide'
        
        elif tipo=='Gaussian':
            correction_bordes='Gaussian'

        elif tipo=='No':
            correction_bordes=None
            
        break
    elif cam=='Zenmuse X5S (DJI Inspire 2)':
        width=5280
        height=2970
        FOVwidth=64.686982896768800
        FOVheight=39.211354832645280
        band=0
        img_type=3
        
        break
    
    elif cam == 'DJI Mavic Pro':
        width=4000
        heigth=2250
        FOVwidth=71.19928287983431
        FOVheight=43.86979719641275
        band=0
        img_type=4
        
        break
    
    elif cam=='Set parameters manually':
        
        parametro='Width (px)'
        width=int(interactive_window_number(parametro))
        
        parametro='Height (px)'
        height=int(interactive_window_number(parametro))
        
        parametro='Field of view in the long side (width)'
        FOVwidth=int(interactive_window_number(parametro))
        
        parametro='Field of view in the short side (height)'
        FOVheight=int(interactive_window_number(parametro))
        
        break
    else:
        print('''

        You have not selected a valid option!!!

        ''')
        pass



k=67
Prefix=prefijo(k)
name=os.path.join(SourcePath, Prefix+str(k)+"_"+str(band)+".tif")
pil_img = Image.open(name)
binary_img = open(name,"rb")
capture_img = capture.Capture.from_filelist([name])



flag_features='d'

img,h12,h32,imReg,height, width,resHeight,resWidth, pt_sup_izq,x1,x2,x3,x4,x_min,y_min,yawDeg = generacion_nueva_img_general(img_type,height, width, FOVwidth, FOVheight,capture_img, pil_img, binary_img,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes)


# flag_features='d'

# img2,h12,h32,imReg2,height, width,resHeight,resWidth, pt_sup_izq,x1,x2,x3,x4,x_min,y_min,yawDeg = generacion_nueva_img_general(img_type,height, width, FOVwidth, FOVheight,capture_img, pil_img, binary_img,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes)
# rgb_image = np.stack((imReg[0], imReg[1], imReg[2]), axis=-1)

fig,ax=plt.subplots(1,2)

# ax[0].set_title('Imagen Original')

ax[0].imshow(img[:,:],label='Imagen Original')
ax[1].imshow(imReg[0],label='Imagen Reconstruida')


# plt.text(100, 100, "Imagen Oblicua", fontsize=20, color='Blue', fontweight='bold', fontfamily='serif')
# plt.savefig("D:/fgalletta/Metodos Experimentales/oblicua1.png")
# plt.figure()
# ax[1].set_title('Imagen Reconstruida')


# plt.figure()
# # ax[1].set_title('Imagen Reconstruida')
# plt.imshow(eq,label='Imagen Reconstruida')


# plt.text(100, 150, "Imagen Reconstruida", fontsize=20, color='Blue', fontweight='bold', fontfamily='serif')
# plt.savefig("D:/fgalletta/Metodos Experimentales/reconstruida1.png")

# plt.figure()
# plt.imshow(img2,cmap='gray',label='Imagen Original')
# plt.text(10, 10, "Imagen Oblicua", fontsize=20, color='Blue', fontweight='bold', fontfamily='serif')
# plt.savefig("D:/fgalletta/Metodos Experimentales/oblicua2.png")
# plt.figure()
# plt.imshow(imReg2[0],cmap='gray',label='Imagen Reconstruida')
# plt.text(10, 10, "Imagen Reconstruida", fontsize=20, color='Blue', fontweight='bold', fontfamily='serif')
# plt.savefig("D:/fgalletta/Metodos Experimentales/reconstruida2.png")
plt.show()