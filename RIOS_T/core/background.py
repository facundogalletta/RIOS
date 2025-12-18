import core.micasense.capture as capture
import numpy as np
from core.correction_distorted_img import calibration_thermal
import PIL.Image as Image

def background(path_imgs,band):
    """Funcion que realiza el background de fotos.
    
    Args: 
        path_imgs (string) : Nombres de las las fotos para realizar background.
    
    Returns:
        array : Background
    """
    width, height=Image.open(path_imgs[0]).size
    
    img_sum=np.zeros((height,width))
    for path in path_imgs:
        capture_img = capture.Capture.from_filelist([path])
        
        img=calibration_thermal(capture_img,band, height, width,DT_promedio_agua_fria=[],correction_bordes=None,panel_names=None,sky_names=None)
        img_sum+=img#.astype('float32')
    img=img_sum/len(path_imgs)
    return img
