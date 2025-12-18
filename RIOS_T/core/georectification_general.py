from __future__ import print_function, unicode_literals
import numpy as np
from PIL import Image
import cv2 as cv
import math
import matplotlib.pyplot as plt

from core.correction_distorted_img import calibration_raw,calibration_reflectance,calibration_thermal

from core.calibration_dji import calibration_dji
# from core.find_vertices import find_vertices
from core.find_vertices_new import find_vertices_new
from core.features import dron_features,camera_dji_features,camera_features


def georef_img(img_type,cam,height, width, FOVwidth, FOVheight,capture_img, pil_img, binary_img,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes,panel_names,sky_names):
    
    """ Genera una nueva imagen rectificada a partir de una imagen original y datos de GPS obtenidos de imagenes o de csv de dron.

    Args:
        height (int) : Altura deseada de la nueva imagen.
        width (int) : Ancho deseado de la nueva imagen.
        FOVwidth (float) : Ángulo de visión de la cámara en dirección horizontal (en grados).
        FOVheight (float) : Ángulo de visión de la cámara en dirección vertical (en grados).
        capture_img (Imagen .capture): Imagen obtenida desde miscasense.capture. Se utiliza para corregir distorsion tambien implementada por micasense
        pil_img (PIL Image) : Imagen en formato PIL.
        binary_img (binary imagen con open()) : Imagen binaria con hora de la toma de la misma.
        band (int) : Banda espectral de la imagen a georectificar.
        csv_file (str): Ruta al archivo CSV que contiene los datos del dron con timestamp.
        flag_features (str): Indicador para determinar si se utilizan características de la cámara ('c') o del dron ('d').
        DT_promedio_agua_fria: Imagen de la variacion de temperatura promedio en imagenes de agua fría.
        correction_bordes: Indica corrección de los bordes a implementar en la imagen.
        sky_names: Nombre de las imagenes del cielo utilizadas para la corrección de reflectancia.

    Returns:
        tuple: Una tupla que contiene la imagen original, las matrices de transformación, la imagen rectificada,
               la altura y el ancho de la nueva imagen, la resolución en altura y ancho en metros/pixel, y las coordenadas del vértice superior izquierdo de la nueva imagen.
    """

    
    # Conversión de FOV a radianes 
    FOVw = math.radians(FOVwidth)  # Ángulo de visión de la cámara en la dirección larga de la foto
    FOVh = math.radians(FOVheight)  # Ángulo de visión de la cámara en la dirección corta de la foto

    # Determinar FOV de la imagen barrida en las atistas de la foto.
  
    if flag_features == 'c':
        yaw, pitch, roll, x, y, z,yawDeg,date_num= camera_features(pil_img, binary_img)
    elif flag_features == 'd':
        yaw, pitch, roll, x, y, z,yawDeg,date_num= dron_features(csv_file, binary_img, desfasaje=0)
    elif flag_features == 'dji':
        yaw, pitch, roll, x, y, z,yawDeg,date_num= camera_dji_features(binary_img)
    # print(yaw, pitch, roll, x, y, z)
    
    # z=z
  
    # Calibrar distorsion y efectos de bordes de la imagen
    
    if cam=='1':
        
        if img_type == 3:  
            img = calibration_reflectance(capture_img,band, height, width,panel_names,sky_names)
            img = np.array(Image.fromarray(img))
        else:
            img = calibration_raw(pil_img,band, height, width,panel_names,sky_names)
            img = np.array(Image.fromarray(img))
        
    elif cam=='2':
        img = calibration_thermal(capture_img,band, height, width,DT_promedio_agua_fria,correction_bordes,panel_names,sky_names)
        img = np.array(Image.fromarray(img))
        
    elif cam=='3' or cam=='4' or cam=='5':
        img = calibration_dji(pil_img,height , width)
        img = np.array(img)
    # Genero vertices ortogonales en el agua (solo con yaw). Genera un cuadrado en la ubicacion x,y,z de la camara orientado segun el yaw.

    pitch0 = 0
    pitch0 = math.radians(pitch0)
    roll0 = 0
    roll0 = math.radians(roll0) 
    
    x1_or, x2_or, x3_or, x4_or = find_vertices_new(x, y, z, yaw, pitch0, roll0, FOVw, FOVh)
    x1, x2, x3, x4 = find_vertices_new(x, y, z, yaw, pitch, roll, FOVw, FOVh)

    # Tranformation UTM to pixeles 
    
    # pts_src = np.array([x2_or, x1_or, x4_or, x3_or])
    # pts_dst = np.array([[width, 0], [0, 0], [width, height], [0, height]])
    
    pts_src = np.array([x1_or, x2_or, x4_or, x3_or])
    pts_dst = np.array([[0, 0], [width, 0], [width, height], [0, height]])
    
    h12, status = cv.findHomography(pts_src, pts_dst)

    # Calcular vertices en pixeles de la imagen ortogonal (deberían caer en [0, 0], [0, width], [height, 0], [height, width]) por ser pts_dst.
    
    x1_or_px = [np.dot(h12, np.array([x1_or[0], x1_or[1], 1]))[0] / np.dot(h12, np.array([x1_or[0], x1_or[1], 1]))[2],
                np.dot(h12, np.array([x1_or[0], x1_or[1], 1]))[1] / np.dot(h12, np.array([x1_or[0], x1_or[1], 1]))[2]]
    
    x2_or_px = [np.dot(h12, np.array([x2_or[0], x2_or[1], 1]))[0] / np.dot(h12, np.array([x2_or[0], x2_or[1], 1]))[2],
                np.dot(h12, np.array([x2_or[0], x2_or[1], 1]))[1] / np.dot(h12, np.array([x2_or[0], x2_or[1], 1]))[2]]
    
    x3_or_px = [np.dot(h12, np.array([x3_or[0], x3_or[1], 1]))[0] / np.dot(h12, np.array([x3_or[0], x3_or[1], 1]))[2],
                np.dot(h12, np.array([x3_or[0], x3_or[1], 1]))[1] / np.dot(h12, np.array([x3_or[0], x3_or[1], 1]))[2]]
    
    x4_or_px = [np.dot(h12, np.array([x4_or[0], x4_or[1], 1]))[0] / np.dot(h12, np.array([x4_or[0], x4_or[1], 1]))[2],
                np.dot(h12, np.array([x4_or[0], x4_or[1], 1]))[1] / np.dot(h12, np.array([x4_or[0], x4_or[1], 1]))[2]]
    
    # Utilizo homografia con coordenada UTM para obtener vertices de la nueva imagen rectificada. Dado que conserva la forma. La resolucion 
    # de esta imagen esta determinada por las coordenadas del cuadrado pitch,roll=0 y los lados de la imagen.
    
    x1_px = [np.dot(h12, np.array([x1[0], x1[1], 1]))[0] / np.dot(h12, np.array([x1[0], x1[1], 1]))[2],
             np.dot(h12, np.array([x1[0], x1[1], 1]))[1] / np.dot(h12, np.array([x1[0], x1[1], 1]))[2]]
    
    x2_px = [np.dot(h12, np.array([x2[0], x2[1], 1]))[0] / np.dot(h12, np.array([x2[0], x2[1], 1]))[2],
             np.dot(h12, np.array([x2[0], x2[1], 1]))[1] / np.dot(h12, np.array([x2[0], x2[1], 1]))[2]]
    
    x3_px = [np.dot(h12, np.array([x3[0], x3[1], 1]))[0] / np.dot(h12, np.array([x3[0], x3[1], 1]))[2],
             np.dot(h12, np.array([x3[0], x3[1], 1]))[1] / np.dot(h12, np.array([x3[0], x3[1], 1]))[2]]
    
    x4_px = [np.dot(h12, np.array([x4[0], x4[1], 1]))[0] / np.dot(h12, np.array([x4[0], x4[1], 1]))[2],
             np.dot(h12, np.array([x4[0], x4[1], 1]))[1] / np.dot(h12, np.array([x4[0], x4[1], 1]))[2]]
    
    # Box de soporte de la nueva imagen
    
    # Se puede definir el factor de escala que reduce la resolución de la imagen rectificada
    fs = 1
    
    x1_px[0]=x1_px[0]/fs
    x1_px[1]=x1_px[1]/fs
    
    x2_px[0]=x2_px[0]/fs
    x2_px[1]=x2_px[1]/fs
    
    x3_px[0]=x3_px[0]/fs
    x3_px[1]=x3_px[1]/fs
    
    x4_px[0]=x4_px[0]/fs
    x4_px[1]=x4_px[1]/fs
    
    # Transformacion de px de imagen oblicua a nueva rectificada. 

    pts_src = np.array([[0, 0], [width, 0], [0, height], [width, height]])
    pts_dst = np.array([ x1_px,x2_px, x3_px, x4_px])
    
    h23, status = cv.findHomography(pts_src, pts_dst)
    
    # La resolucion de la imagen rectificada queda determinada por la construccion de la homografia que lleva el cuadrado de find_vertices
    # para pitch y roll = 0, a el cuadrado dado por (height,width)

    ancho=2*z*math.tan(FOVw/2) #(m)
    altura=2*z*math.tan(FOVh/2) #(m)
    resWidth=ancho/width*fs#(m/px)
    resHeight=altura/height*fs #(m/px)

    # Guardo inversa, util para despues pasar de un punto en el agua a su valor en pixel en la imagen oblicua
    h32=np.linalg.inv(h23)

    # Calcular los puntos extremos de la imagen de entrada después de la transformación
    pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    transformed_pts = cv.perspectiveTransform(np.array([pts]), h23)

    # Calcular las dimensiones de la imagen de salida
    x_min = np.min(transformed_pts[:, :, 0])
    x_max = np.max(transformed_pts[:, :, 0])
    y_min = np.min(transformed_pts[:, :, 1])
    y_max = np.max(transformed_pts[:, :, 1])
    w_out = int(x_max - x_min)
    h_out = int(y_max - y_min)
    
    # Los pixeles rectificados no caen en el box generado desde (0,0), dado que comienza en xmin e ymin.
    
    h23_trasl = np.array([[1, 0, -np.floor(x_min)], [0, 1, -np.floor(y_min)], [0, 0, 1]])
    # h23_trasl = np.array([[1, 0, -(pts_dst.min(axis=0)[1])], [0, 1, -(pts_dst.min(axis=0)[0])], [0, 0, 1]])
    
    h_final = np.dot(h23_trasl, h23)
    
    imReg=[]
    if len(img.shape) == 3:
        for i in range(3):
            im = img[:,:,int(i)]
            # Relleno la imagen nueva con la tranformacion y traslacion determinada.
            imReg.append(cv.warpPerspective(im, h_final, (w_out, h_out)))
    else:
        imReg.append(cv.warpPerspective(img, h_final,(w_out, h_out)))
    
    
    # Obtener coordenadas de vertices UTM de la nueva imagen, no necesariamente corresponde a un punto de la imagen rectifiada.

    h21 = np.linalg.inv(h12)

    # Vertice superior izquierdo real
    pt_sup_izq = [np.dot(h21, np.array([x_min*fs, y_min*fs, 1]))[0] / np.dot(h21, np.array([x_min*fs, y_min*fs, 1]))[2],
                  np.dot(h21, np.array([x_min*fs, y_min*fs, 1]))[1] / np.dot(h21, np.array([x_min*fs, y_min*fs, 1]))[2]]
    
    return img, h12, h32, imReg, h_out, w_out, resHeight, resWidth, pt_sup_izq, x1, x2, x3, x4, x_min, y_min, yawDeg,date_num,h_final,x,y,z
