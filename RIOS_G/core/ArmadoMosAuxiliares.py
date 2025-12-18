# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 14:29:07 2020

@author: maurol
"""

##############################################################################
################# ARMADO DE MOSAICO - CÁMARA MULTIESPECTRAL ##################
#################            Funciones Auxiliares           ##################
##############################################################################

# =============================================================================#
""" Función para encontrar Yaw, Pitch y Roll del string con los metadatos """


def orientacion(STRwithMetadata):
    i = 1
    find = 0
    while i < len(STRwithMetadata) and find == 0:
        if STRwithMetadata[i:(i + 20)] == 'Camera:IrradianceYaw':
            find = 1
            start = i + 21
            find_end = 0
            k = 2
            while find_end == 0:
                if STRwithMetadata[i + 21 + k] == '<':
                    end = i + 21 + k - 1
                    find_end = 1
                else:
                    k = k + 1
            YAW_in_Deg = float(STRwithMetadata[start:end])
        else:
            i = i + 1
    i = 1
    find = 0
    while i < len(STRwithMetadata) and find == 0:
        if STRwithMetadata[i:(i + 22)] == 'Camera:IrradiancePitch':
            find = 1
            start = i + 23
            find_end = 0
            k = 2
            while find_end == 0:
                if STRwithMetadata[i + 23 + k] == '<':
                    end = i + 23 + k - 1
                    find_end = 1
                else:
                    k = k + 1
            PITCH_in_Deg = float(STRwithMetadata[start:end])
        else:
            i = i + 1
    i = 1
    find = 0
    while i < len(STRwithMetadata) and find == 0:
        if STRwithMetadata[i:(i + 21)] == 'Camera:IrradianceRoll':
            find = 1
            start = i + 22
            find_end = 0
            k = 2
            while find_end == 0:
                if STRwithMetadata[i + 22 + k] == '<':
                    end = i + 22 + k - 1
                    find_end = 1
                else:
                    k = k + 1
            ROLL_in_Deg = float(STRwithMetadata[start:end])
        else:
            i = i + 1
    return [YAW_in_Deg, PITCH_in_Deg, ROLL_in_Deg]


###############################################################################

# =============================================================================#
""" Funciones para encontrar posición (GPS) """


def str2lat_or_long_vec(string):
    deg = string[1:3]
    minutes = string[5:7]
    sec = string[9:len(string) - 1]

    coord = [float(deg), float(minutes), str2fraction(sec)]

    return coord


def LongLat2UTM(long, lat, formato):
    # formato: 1 si long y lat vienen como lista [gr,min,seg] y con signo positivo o 2 si vienen como grados decimales con signo negativo

    # from pyproj import Proj
    import numpy as np

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


def str2fraction(string):
    for i in range(len(string)):
        if string[i] == '/':
            indDiv = i
    try:  # Significa que string vino como un número entero
        num = string[0:indDiv]
        den = string[indDiv + 1:len(string)]
        num = float(num)
        den = float(den)

        fraction = num / den
    except:
        fraction = float(string)

    return fraction


def datetimeSTR2timeFLOAT(datetime, subsec):
    hour = int(datetime[11:13])
    minute = int(datetime[14:16])
    seconds = int(datetime[17:19])
    timeFLOAT = float(hour * 10000 + minute * 100 + seconds)
    subsecFLOAT = float('0.' + subsec)
    timeFLOAT = timeFLOAT + subsecFLOAT
    return timeFLOAT


def fracc_sec(timeColDes, tickCol):
    sec_actual = timeColDes[0]
    timeColDes[0] = timeColDes[0] + 0.05
    for i in range(1, len(timeColDes)):
        if timeColDes[i] == sec_actual:
            timeColDes[i] = timeColDes[i - 1] + (tickCol[i] - tickCol[i - 1]) / 4500000
        else:
            sec_actual = timeColDes[i]
            timeColDes[i] = timeColDes[i] + 0.05
    return timeColDes
