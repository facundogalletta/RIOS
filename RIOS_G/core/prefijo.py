def prefijo(k,prefix):
    """ Funcion para obtener el prefijo de las fotos tomadas.

    Args:
        k: Numero de foto
        prefix: Nombre definido segun la camara
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