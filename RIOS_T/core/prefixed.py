
'''
Author: fgalltta@fing.edu.uy
Date: 3/20/2024
Description: This code obtains the prefix of the photos for each camera used.
'''

###############################################
# Function to obtain the prefix of the photos #
###############################################

def prefijo(k,prefix):
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
