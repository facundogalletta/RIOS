# from pyproj import Proj
import numpy as np
    
def LongLat2UTM(long, lat, formato):
    """
    Convert longitude and latitude to UTM coordinates
    
    input:
        long: longitude in degrees
        lat: latitude in degrees
        formato: 1 si long y lat vienen como lista [gr,min,seg] y con signo positivo o 2 si vienen como grados decimales con signo negativo
    
    output:
        x: x coordinate in UTM
        y: y coordinate in UTM
    """
    
    if formato == 1:
        lo = long[0] + long[1] / 60 + long[2] / 3600
        lo = -lo

        la = lat[0] + lat[1] / 60 + lat[2] / 3600
        la = -la
    else:
        lo = long
        la = lat

    # print('Coordenadas foto (long,lat en grados decimales):',str(lo),' , ',str(la))

    # myProj = Proj("+proj=utm +zone=21H, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    # x1,y1=myProj(lo,la)

    sa = 6378137.000000
    sb = sb = 6356752.314245
    e2 = (((sa ** 2) - (sb ** 2)) ** 0.5) / sb
    e2cuadrada = e2 ** 2
    c = (sa ** 2) / sb

    Lat = la * np.pi / 180
    Lon = lo * np.pi / 180

    Huso = np.fix((lo / 6) + 31)
    S = ((Huso * 6) - 183)
    deltaS = Lon - (S * np.pi / 180)

    a = np.cos(Lat) * np.sin(deltaS)
    epsilon = 0.5 * np.log((1 + a) / (1 - a))
    nu = np.arctan(np.tan(Lat) / np.cos(deltaS)) - Lat
    v = (c / ((1 + (e2cuadrada * (np.cos(Lat)) ** 2))) ** 0.5) * 0.9996
    ta = (e2cuadrada / 2) * epsilon ** 2 * (np.cos(Lat)) ** 2
    a1 = np.sin(2 * Lat)
    a2 = a1 * (np.cos(Lat)) ** 2
    j2 = Lat + (a1 / 2)
    j4 = ((3 * j2) + a2) / 4
    j6 = ((5 * j4) + (a2 * (np.cos(Lat)) ** 2)) / 3
    alfa = (3 / 4) * e2cuadrada
    beta = (5 / 3) * alfa ** 2
    gama = (35 / 27) * alfa ** 3
    Bm = 0.9996 * c * (Lat - alfa * j2 + beta * j4 - gama * j6)

    x = epsilon * v * (1 + (ta / 3)) + 500000
    y = nu * v * (1 + ta) + Bm

    if y < 0:
        y = 9999999 + y

    return x, y