
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# from Band_alignment import band_alignment
import core.micasense.capture as capture
from core.georectification_general import georef_img


def intersection(img_type,cam,path_img1,path_img2, FOVwidth,FOVheight,band,csv_file,flag_features, flag_interseccion,DT_promedio_agua_fria,correction_bordes,panel_names,sky_names):
    """ Calcula la intersección entre dos polígonos definidos por el campo de vision del agua de dos imágenes y devuelve información relacionada.

    Parameters:
        path_img1 (str): Ruta de la primera imagen original.
        path_img2 (str): Ruta de la segunda imagen original.
        FOVwidth (float): Ancho del campo de visión de las imagens.
        FOVheight (float): Alto del campo de visión de las imagens.
        band (int): Número de banda espectarl de las imagenes.
        csv_file (str): Ruta del archivo CSV del con datos del vuelo del tron y timestamp.
        flag_features (bool): Bandera para indicar de donde obtener parametros del vuelo del dron.
        flag_interseccion (bool): Bandera para indicar si se muestra la intersección.
        DT_promedio_agua_fria (float): Imagen promedio de agua fría utilizada para correccion de efectos de borde.
        correction_bordes (string): indicar el tipo de corrección de bordes.
        test (Optional[bool]): Opcional. Valor para indicar si es prueba o iteracion.

    Returns:
        tuple: Una tupla que contiene varias matrices y coordenadas relacionadas.

    """

    # Para ambas imagenes a intesectar:
    # Determinar imagen original y rectificada (y su resolucion y resolucion px/m) sin distorsion ni efectos de borde y coordenadas.
    # Se obtienen tambien coordenadas UTM de las esquinas de la imagen en el agua 
 
    pil_img1 = Image.open(path_img1)
    binary_img1 = open(path_img1, "rb")
    
    if cam=='1' or cam=='2':
        capture_img1 = capture.Capture.from_filelist([path_img1])
    else:
        capture_img1 = []
        
    height=pil_img1.size[1]
    width=pil_img1.size[0]
    
    res_georef_1=georef_img(img_type,cam,height, width, FOVwidth, FOVheight, capture_img1, pil_img1, binary_img1,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes,panel_names,sky_names)
    img1_undist=res_georef_1[0]
    h12_1=res_georef_1[1]
    h32_1=res_georef_1[2]
    im1Rec=res_georef_1[3]
    height_rec_1=res_georef_1[4]
    width_rec_1=res_georef_1[5]
    x1_1=res_georef_1[9]
    x2_1=res_georef_1[10]
    x3_1=res_georef_1[11]
    x4_1=res_georef_1[12]
    x_min1=res_georef_1[13]
    y_min1=res_georef_1[14]
    h_final_1=res_georef_1[17]
    h_final_1_1=np.linalg.inv(h_final_1)
    X_1=res_georef_1[18]
    Y_1=res_georef_1[19]
    Z_1=res_georef_1[20]
    
    pil_img2 = Image.open(path_img2)
    binary_img2 = open(path_img2, "rb")
    if cam=='1' or cam=='2':
        capture_img2 = capture.Capture.from_filelist([path_img2])
    else:
        capture_img2 = []
    
    res_georef_2=georef_img(img_type,cam,height, width, FOVwidth, FOVheight, capture_img2, pil_img2, binary_img2,band, csv_file,flag_features,DT_promedio_agua_fria,correction_bordes,panel_names,sky_names)
    img2_undist=res_georef_2[0]
    h12_2=res_georef_2[1]
    h32_2=res_georef_2[2]
    im2Rec=res_georef_2[3]
    height_rec_2=res_georef_2[4]
    width_rec_2=res_georef_2[5]
    x1_2=res_georef_2[9]
    x2_2=res_georef_2[10]
    x3_2=res_georef_2[11]    
    x4_2=res_georef_2[12]
    x_min2=res_georef_2[13]
    y_min2=res_georef_2[14]
    h_final_2=res_georef_2[17]
    h_final_2_1=np.linalg.inv(h_final_2)
    X_2=res_georef_2[18]
    Y_2=res_georef_2[19]
    Z_2=res_georef_2[20]
    
    
   # Definir los polígonos del campo de vision de ambas imagenes en el agua
   
    poligono1 = Polygon([(x1_1[0],x1_1[1]), (x3_1[0],x3_1[1]), (x4_1[0],x4_1[1]),(x2_1[0],x2_1[1])]) 
    poligono2 = Polygon([(x1_2[0],x1_2[1]), (x3_2[0],x3_2[1]), (x4_2[0],x4_2[1]),(x2_2[0],x2_2[1])])

    # Calcular la intersección en el agua de los dos poligonos.

    interseccion = poligono1.intersection(poligono2)

    if interseccion.is_empty:
        print("Los polígonos no se intersectan")
        x_interseccion, y_interseccion=[],[]
    else:
        x_interseccion, y_interseccion = interseccion.exterior.xy
            
    px_img1=[]
    px_img2=[]
    px_img1r=[]
    px_img2r=[]

    for i in range(len(x_interseccion)):
        x=x_interseccion[i]
        y=y_interseccion[i]


        # Ir de las coordenadas reales a coordenadas en cada imagen. Se utilizan homografias descriptas en generacion_nueva_img.
        
        # Obtener coordenadas en pixeles de imagenes rectificadas de los vertices de la intereccion.

        p1_px_r = [np.dot(h12_1, np.array([x, y, 1]))[0] / np.dot(h12_1, np.array([x, y, 1]))[2],
                   np.dot(h12_1, np.array([x, y, 1]))[1] / np.dot(h12_1, np.array([x, y, 1]))[2]]
        
        p2_px_r = [np.dot(h12_2, np.array([x, y, 1]))[0] / np.dot(h12_2, np.array([x, y, 1]))[2],
                   np.dot(h12_2, np.array([x, y, 1]))[1] / np.dot(h12_2, np.array([x, y, 1]))[2]]
        
        p1_px_r[0] = p1_px_r[0]-x_min1
        p1_px_r[1] = p1_px_r[1]-y_min1
        
        p2_px_r[0] = p2_px_r[0]-x_min2
        p2_px_r[1] = p2_px_r[1]-y_min2
        
        p1_px = [np.dot(h_final_1_1, np.array([p1_px_r[0], p1_px_r[1], 1]))[0] / np.dot(h_final_1_1, np.array([p1_px_r[0], p1_px_r[1], 1]))[2],
                 np.dot(h_final_1_1, np.array([p1_px_r[0], p1_px_r[1], 1]))[1] / np.dot(h_final_1_1, np.array([p1_px_r[0], p1_px_r[1], 1]))[2]]
        
        p2_px=[np.dot(h_final_2_1, np.array([p2_px_r[0], p2_px_r[1], 1]))[0] / np.dot(h_final_2_1, np.array([p2_px_r[0], p2_px_r[1], 1]))[2],
               np.dot(h_final_2_1, np.array([p2_px_r[0], p2_px_r[1], 1]))[1] / np.dot(h_final_2_1, np.array([p2_px_r[0], p2_px_r[1], 1]))[2]]
        
        # # Obtengo coordenadas en pixeles de imagenes oblicuas de los vertices de la interseccion.

        # p1_pxp = [np.dot(h32_1, np.array([p1_px[0], p1_px[1], 1]))[0] / np.dot(h32_1, np.array([p1_px[0], p1_px[1], 1]))[2],
        #           np.dot(h32_1, np.array([p1_px[0], p1_px[1], 1]))[1] / np.dot(h32_1, np.array([p1_px[0], p1_px[1], 1]))[2]]
        # p2_pxp = [np.dot(h32_2, np.array([p2_px[0], p2_px[1], 1]))[0] / np.dot(h32_2, np.array([p2_px[0], p2_px[1], 1]))[2],
        #           np.dot(h32_2, np.array([p2_px[0], p2_px[1], 1]))[1] / np.dot(h32_2, np.array([p2_px[0], p2_px[1], 1]))[2]]
        
        px_img1r.append(p1_px_r)
        px_img2r.append(p2_px_r)
        px_img1.append(p1_px)
        px_img2.append(p2_px)
    
    px_img1=np.array(px_img1)
    px_img2=np.array(px_img2)
    px_img1r=np.array(px_img1r)
    px_img2r=np.array(px_img2r)

    #Corrijo vertices que caen afuera de imagen oblicua por error de estimaciones usando las homografias h12 y h32

    for i in range(px_img1.shape[0]):
        if px_img1[i,1]<0:
            px_img1[i,1]=0
        elif px_img1[i,1]>height:
             px_img1[i,1]=height
        
        if px_img1[i,0]<0:
            px_img1[i,0]=0
        elif px_img1[i,0]>width:
             px_img1[i,0]=width

    for i in range(px_img2.shape[0]):
        if px_img2[i,1]<0:
            px_img2[i,1]=0
        elif px_img2[i,1]>height:
             px_img2[i,1]=height
        
        if px_img2[i,0]<0:
            px_img2[i,0]=0
        elif px_img2[i,0]>width:
             px_img2[i,0]=width

    #Corrijo vertices que caen afuera de imagen rectificada por error de estimaciones usando las homografias h12.

    for i in range(px_img1r.shape[0]):
        if px_img1r[i,1]<0:
            px_img1r[i,1]=0
        elif px_img1r[i,1]>height_rec_1:
             px_img1r[i,1]=height_rec_1
        
        if px_img1r[i,0]<0:
            px_img1r[i,0]=0
        elif px_img1r[i,0]>width_rec_1:
             px_img1r[i,0]=width_rec_1

    for i in range(px_img2r.shape[0]):
        if px_img2r[i,1]<0:
            px_img2r[i,1]=0
        elif px_img2r[i,1]>height_rec_2:
             px_img2r[i,1]=height_rec_2
        
        if px_img2r[i,0]<0:
            px_img2r[i,0]=0
        elif px_img2r[i,0]>width_rec_2:
             px_img2r[i,0]=width_rec_2
    
    if flag_interseccion==True:

        #Ploteo poligonos en coordenadas reales e interseccion:

        fig, ax = plt.subplots()
        plt.gca().invert_yaxis()
        ax.plot(*poligono1.exterior.xy, label='Polígono 1')
        ax.plot(*poligono2.exterior.xy, label='Polígono 2')
        ax.plot(x_interseccion, y_interseccion, 'r', label='Intersección')
        ax.legend()

        #Coordenadas de interseccion en fotos oblicuas:

        fig, ax = plt.subplots()
        ax.imshow(img1_undist)
        ax.plot(px_img1[:,0], px_img1[:,1], 'r',linewidth=3)

        # Quitar ejes y bordes por completo
        ax.axis('off')

        # Quitar margen alrededor (importante)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Guardar sin borde ni nada agregado
        plt.savefig(
            path_img1.split('.')[0] + '_int_con_' + path_img2.split('\\')[-1].split('.')[0] + '.png',
            bbox_inches='tight',
            pad_inches=0
        )
        
        fig, ax = plt.subplots()
        ax.imshow(img2_undist)
        ax.plot(px_img2[:,0], px_img2[:,1], 'r',linewidth=3)

        # Quitar ejes y bordes por completo
        ax.axis('off')

        # Quitar margen alrededor (importante)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Guardar sin borde ni nada agregado
        plt.savefig(
            path_img2.split('.')[0] + '_int_con_' + path_img1.split('\\')[-1].split('.')[0] + '.png',
            bbox_inches='tight',
            pad_inches=0
        )
        
        #Coordenadas de interseccion en fotos rectificadas:

        plt.figure()
        plt.imshow(im1Rec[0],cmap='gray')
        plt.plot(px_img1r[:,0],px_img1r[:,1],'b',label=' Interseccion de imagen 2 con 1')
        # plt.legend(loc='center')
        # plt.savefig('./csic_ute_imageprocessing-master/prueba/res_int/'+'img1_rec_con_regionint.png')

        plt.figure()
        plt.imshow(im2Rec[0],cmap='gray')
        plt.plot(px_img2r[:,0],px_img2r[:,1],'b',label='Interseccion de imagen 1 con 2')
        # plt.legend(loc='center')
        # plt.savefig('./csic_ute_imageprocessing-master/prueba/res_int/'+'img2_rec_con_regionint.png')
        
        
        if False:
        
            origin_1=np.array([X_1,Y_1,Z_1])
            origin_2=np.array([X_2,Y_2,Z_2])
            
            fig = plt.figure()
            ax3d = fig.add_subplot(111, projection='3d')

            # Polígono proyectado
            ax3d.plot([x1_1[0], x2_1[0]], [x1_1[1], x2_1[1]], [0, 0], color='b', label='Capture 1')
            ax3d.plot([x2_1[0], x4_1[0]], [x2_1[1], x4_1[1]], [0, 0], color='b')
            ax3d.plot([x4_1[0], x3_1[0]], [x4_1[1], x3_1[1]], [0, 0], color='b')
            ax3d.plot([x3_1[0], x1_1[0]], [x3_1[1], x1_1[1]], [0, 0], color='b')
            
            # Polígono proyectado
            ax3d.plot([x1_2[0], x2_2[0]], [x1_2[1], x2_2[1]], [0, 0], color='orange', label='Capture 2')
            ax3d.plot([x2_2[0], x4_2[0]], [x2_2[1], x4_2[1]], [0, 0], color='orange')
            ax3d.plot([x4_2[0], x3_2[0]], [x4_2[1], x3_2[1]], [0, 0], color='orange')
            ax3d.plot([x3_2[0], x1_2[0]], [x3_2[1], x1_2[1]], [0, 0], color='orange')
            

            ax3d.plot(x_interseccion, y_interseccion,[0]*len(x_interseccion), 'r', label='Intersecction')
            
            
            # Conexiones cámara-esquinas
            ax3d.plot([origin_1[0],x1_1[0]], [origin_1[1],x1_1[1]], [origin_1[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_1[0],x2_1[0]], [origin_1[1],x2_1[1]], [origin_1[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_1[0],x3_1[0]], [origin_1[1],x3_1[1]], [origin_1[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_1[0],x4_1[0]], [origin_1[1],x4_1[1]], [origin_1[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            
            # Conexiones cámara-esquinas
            ax3d.plot([origin_2[0],x1_2[0]], [origin_2[1],x1_2[1]], [origin_2[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_2[0],x2_2[0]], [origin_2[1],x2_2[1]], [origin_2[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_2[0],x3_2[0]], [origin_2[1],x3_2[1]], [origin_2[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            ax3d.plot([origin_2[0],x4_2[0]], [origin_2[1],x4_2[1]], [origin_2[2], 0], color='m', linestyle='dotted',linewidth=0.5)
            
            
            font = {'fontname':'Times New Roman', 'fontweight':'bold', 'fontsize':16}
            
            ax3d.set_xlabel('X', **font)
            ax3d.set_ylabel('Y', **font)
            ax3d.set_zlabel('Z', **font)
            
            ax3d.view_init(elev=20, azim=-40)
            ax3d.legend(fontsize=8, loc='upper right')
            ax3d.zaxis.set_pane_color((.8, 0.9, 1.0, 1.0))
            ax3d.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

            plt.savefig('./Figs/intersections.png', dpi=500, bbox_inches='tight')
            plt.show()
            
    # Obtener coordenadas en pixeles de imagenes oblicuas del baricentro de la interseccion de las imagenes oblicuas en el agua.

    interseccion_xy=list(zip(x_interseccion, y_interseccion))
    pligono_interseccion=Polygon(interseccion_xy)
    baricentro=pligono_interseccion.centroid
    x, y = baricentro.x, baricentro.y
    
    baricentro_1=[]
    baricentro_2=[]
    
    bar1_px = [np.dot(h12_1, np.array([y, x, 1]))[0] / np.dot(h12_1, np.array([y, x, 1]))[2],
                np.dot(h12_1, np.array([y, x, 1]))[1] / np.dot(h12_1, np.array([y, x, 1]))[2]]
    bar2_px = [np.dot(h12_2, np.array([y, x, 1]))[0] / np.dot(h12_2, np.array([y, x, 1]))[2],
                np.dot(h12_2, np.array([y, x, 1]))[1] / np.dot(h12_2, np.array([y, x, 1]))[2]]

    bar1_pxp = [np.dot(h32_1, np.array([bar1_px[0], bar1_px[1], 1]))[0] / np.dot(h32_1, np.array([bar1_px[0], bar1_px[1], 1]))[2],
                np.dot(h32_1, np.array([bar1_px[0], bar1_px[1], 1]))[1] / np.dot(h32_1, np.array([bar1_px[0], bar1_px[1], 1]))[2]]
    bar2_pxp = [np.dot(h32_2, np.array([bar2_px[0], bar2_px[1], 1]))[0] / np.dot(h32_2, np.array([bar2_px[0], bar2_px[1], 1]))[2],
                np.dot(h32_2, np.array([bar2_px[0], bar2_px[1], 1]))[1] / np.dot(h32_2, np.array([bar2_px[0], bar2_px[1], 1]))[2]]
    

    return px_img1, px_img2 , px_img1r, px_img2r,bar1_pxp,bar2_pxp,im2Rec,img1_undist,img2_undist
