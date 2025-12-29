import numpy as np
import matplotlib.pyplot as plt

def int_vector(v,h):
    
    # Definir el plano
    n = np.array([0, 0, 1])  # Vector normal al plano xy
    d = h  # Coeficiente d del plano ax + by + cz + d = 0

    # Calcular la coordenada de intersección (t)
    denominator = np.dot(n, v)
    
    if denominator != 0:
        t = - d / denominator

        # Calcular las coordenadas de la intersección
        interseccion = v * t
    else:
        print("El vector es paralelo al plano, no hay intersección única.")
    
    # interseccion[0] = -interseccion[0]
            
    return interseccion

def find_corners(x, y, z, yaw, pitch, roll, FOVw, FOVh):
    
    """ FUncion que determina los vertices de la imagen en el agua.

    Args:
        x (_type_): coorenada x UTM de la camara.
        y (_type_): coordenada y UTM de la camara.
        z (_type_): coordenada z UTM de la camara. 
        yaw (_type_): orientación de la camara respecto a eje vertical
        pitch (_type_): orientación de la camara respecto a eje local. Cabeceo.
        roll (_type_): orientación de la camara respecto a eje local. Alabeo.
        FOVw (_type_): FOV en el lado largo.
        FOVwPrima (_type_): FOV en el lado largo barrido en la arista de la imagen, determinado con trigonometria.
        FOVh (_type_): FOV en el lado corto.
        FOVhPrima (_type_): FOV en el lado corto barrido en la arista de la imagen, determinado con trigonometria.
        YawVec (_type_): Vector unitario que apunta hacia el yaw en coordenadas UTM.
        BordeVec (_type_): Vector unitario que apunta hacia el borde horizontal de la foto en coordenadas UTM.

    Returns:
        lista: Lista de coordenadas de los vertices de la imagen en UTM. Vertice superior izquierdo, superior derecho, inferior izquierdo e inferior derecho; en ese orden.
    """
    
    # Determinar vertices de la imagen sin rotación 
    
    y1= z * np.tan(FOVh/2)
    x1=-z * np.tan(FOVw/2)

    y2= y1.copy()
    x2= -x1.copy()

    y3=-y1.copy()
    x3=x1.copy()

    y4=-y1.copy()
    x4=-x1.copy()

    # Determinar vectores que apuntan en dirección a los vertices de la imagen sin rotación
    
    v1=np.array([x1,y1,-z])
    v2=np.array([x2,y2,-z])
    v3=np.array([x3,y3,-z])
    v4=np.array([x4,y4,-z])

    ######################
    # MATRIZ DE ROTACIÓN #
    ######################

    CP=np.cos(pitch)
    SP=np.sin(pitch)
    CR=np.cos(roll)
    SR=np.sin(roll)
    CY=np.cos(-yaw)
    SY=np.sin(-yaw)

    M1=np.array([[CY,SY,0], [-SY,CY,0], [0,0,1]])
    M2 = np.array([[1, 0, 0], [0,CP,-SP], [0, SP, CP]])
    M3 = np.array([[CR, 0, SR], [0, 1, 0], [-SR, 0, CR]])
    M=M1.dot(M2.dot(M3))

    v_1=np.array([1,0,0])
    v_2=np.array([0,1,0])
    v_3=np.array([0,0,1])

    vd_1=np.dot(M1,v_1)
    vd_2=np.dot(M1,v_2)
    vd_3=np.dot(M1,v_3)
    
    ####################
    # VECTORES ROTADOS #
    ####################

    v1_p=np.dot(M,v1)
    v2_p=np.dot(M,v2)
    v3_p=np.dot(M,v3)
    v4_p=np.dot(M,v4)

    # Detereminar la intersección con el agua de los vectores rotados, me definen el plano visible en el agua
    
    int_1=int_vector(v1_p,z)
    int_2=int_vector(v2_p,z)
    int_3=int_vector(v3_p,z)
    int_4=int_vector(v4_p,z)

    x1=x+int_1[0]
    y1=y+int_1[1]
    x2=x+int_2[0]
    y2=y+int_2[1]
    x3=x+int_3[0]
    y3=y+int_3[1]
    x4=x+int_4[0]
    y4=y+int_4[1]
        
    return [x1, y1], [x2, y2], [x3, y3], [x4, y4]