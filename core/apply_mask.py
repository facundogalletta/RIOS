import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(I_np,px):

    """Funcion que aplica una mascara de coordenadas en pixeles a una imagen.

    Args:
        I_np (array) : Imagen a la cual se le aplica la mascara.
        px   (array) : Coordenadas de pixeles [[x1,y1],....,[xn,yn]] de la mascara en sentido ant horario.
        
    Returns:
        Ir   (array) : Imagen con la mascara de pixeles.
    """
    
    height, width = I_np.shape
    mask = np.zeros((height, width), dtype=np.uint8)
    cv.fillPoly(mask, [px.astype(np.int32)], color=255)
    
    Ir = cv.bitwise_and(I_np,I_np, mask=mask)
    
    return Ir