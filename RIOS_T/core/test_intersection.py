import glob
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from interseccion_fotos import intersection
import os
from correction_distorted_img import calibration
from micasense import capture
from interactive_windows import interactive_window, interactive_window_number,seleccionar_carpeta,seleccionar_csv,interactive_window_aviso,interactive_window_numbers



width=2015#2064 #(px)
height=1496#1544 #(px)
FOVwidth=48 #(°)
FOVheight=37 #(°)
band=1
img_type=1

SourcePath=seleccionar_carpeta('Path of original images')   
csv_file=seleccionar_csv('Path of csv file with coordinates of images obtained from the drone')

path_imgs = glob.glob(os.path.join(SourcePath, '*.tif'))

def background(path_imgs):
    width, height=Image.open(path_imgs[0]).size
    img_sum=np.zeros((height,width))
    for path in path_imgs:
        capture_img1 = capture.Capture.from_filelist([path])
        img=calibration(capture_img1, band,height,width,DT_promedio_agua_fria=[],correction_bordes=None)
        img_sum+=img#.astype('float32')
    img=img_sum/len(path_imgs)
    return img

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

path_img0 = path_imgs[0] 
path_img1 = path_imgs[1]

flag_features = 'd'

px_img1, px_img2 , px_img1r, px_img2r,im2Rec,height1,width1,height2,width2 = intersection(img_type,path_img0,path_img1,FOVwidth,FOVheight,band,csv_file,flag_features, flag_interseccion = True, DT_promedio_agua_fria=DT_promedio_agua_fria,correction_bordes=correction_bordes)

# imagen_np0 = cv.multiply(np.array(Image.open(path_img0)),1) 
# mask0 = np.zeros((height, width), dtype=np.uint8)
# cv.fillPoly(mask0, [px_img1.astype(np.int64)], color=255)
# imagen_region0 = cv.bitwise_and(imagen_np0,imagen_np0, mask=mask0)

# imagen_np1 = cv.multiply(np.array(Image.open(path_img1)),1) 
# mask1 = np.zeros((height, width), dtype=np.uint8)
# cv.fillPoly(mask1, [px_img2.astype(np.int64)], color=255)
# imagen_region1 = cv.bitwise_and(imagen_np1,imagen_np1, mask=mask1)


# pts_src = np.array(px_img1)
# pts_dst = np.array(px_img2)
# h, status = cv.findHomography(pts_src, pts_dst)

# imtransf = cv.warpPerspective(np.array(Image.open(path_img0)), h, (width, height))

# plt.figure()
# plt.imshow(imtransf,vmin=29200)
# plt.plot(px_img2[:,0],px_img2[:,1],'b')

# plt.figure()
# plt.imshow(Image.open(path_img1))
# plt.plot(px_img2[:,0],px_img2[:,1],'b')


# im_np1=np.array(imtransf)
# img_1reg = cv.bitwise_and(im_np1,im_np1, mask=mask1)

# intensidades1 = img_1reg.flatten().astype(float)
# intensidades2 = imagen_region1.flatten().astype(float)

# plt.figure()
# plt.scatter(intensidades1[intensidades1>29000], intensidades2[intensidades1>29000], marker='.', color='black')

# x=intensidades1[intensidades1>29000]
# y=intensidades2[intensidades1>29000]

# # Realizar el ajuste lineal
# coeficientes = np.polyfit(x, y, 1)
# pendiente = coeficientes[0]
# intercepto = coeficientes[1]

# # Generar los valores predichos
# y_pred = np.polyval(coeficientes, x)

# # Mostrar los resultados
# print("Pendiente:", pendiente)
# print("Intercepto:", intercepto)
plt.show()