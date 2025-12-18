from datetime import datetime, timedelta

def str2datenum(fecha_str: str) -> float:
    """
    Convierte un string tipo "2024:07:17 12:00:56" a datenum estilo MATLAB.
    """
    # Parseo del string
    dt = datetime.strptime(fecha_str, "%Y:%m:%d %H:%M:%S")
    
    # Epoch de MATLAB (0-Jan-0000). En Python no existe el año 0,
    # así que usamos 1-Jan-0001 y sumamos 366 días de offset.
    matlab_origin = datetime(1, 1, 1)
    offset = timedelta(days=365)  # diferencia entre 0000-01-00 y 0001-01-01
    
    delta = dt - matlab_origin + offset
    return delta.days + delta.seconds / 86400 + delta.microseconds / 86400e6
