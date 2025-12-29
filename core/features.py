import math
import pandas as pd
import exifread
import numpy as np
from scipy.interpolate import interp1d

from orientation import orientacion
from str2lat_or_long_vec import str2lat_or_long_vec
from LongLat2UTM import LongLat2UTM
from str2fraction import str2fraction
from datetimeSTR2timeFLOAT import datetimeSTR2timeFLOAT
from fracc_sec import fracc_sec
from strdate2datenum import str2datenum

def dron_features(csv_file, binary_img, desfasaje=0):

    """ Extrae posicion y orientacion del dron a partir de un archivo CSV y una imagen binaria.

    Args:
        csv_file (str): Ruta al archivo CSV que contiene los datos del dron con timestamp.
        binary_img (imagen binaria): Ruta a la imagen binaria con hora de la foto.
        desfasaje (int, opcional): Desfase del reloj en segundos. Por defecto es 0.

    Returns:
        tuple: Una tupla que contiene el yaw, pitch, roll, coordenada x, coordenada y, altitud y yaw en grados. Del dron en el momento de la foto.
    """

    # Leer el archivo CSV y extraer las columnas relevantes

    excel = pd.read_csv(csv_file,
                        skiprows=range(1, 8),
                        usecols=[0, 2, 3, 6, 9, 10, 11, 76])  # En skiprows se pone cant. de filas a sacar sin incluir headings (si se quieren sacar 7 filas poner range(1,8))
    # excel = pd.read_csv(SourcePath+nameXLS)#Excel con los datos
    tickCol = excel['Clock:Tick#']  # Columna del Excel con tickNo
    yawCol = excel['IMU_ATTI(0):yaw:C']  # Columna del Excel con yaw
    pitchCol = excel['IMU_ATTI(0):pitch:C']  # Columna del Excel con pitch
    rollCol = excel['IMU_ATTI(0):roll:C']  # Columna del Excel con roll
    latCol = excel['IMU_ATTI(0):Latitude']  # Columna del Excel con latitud
    lonCol = excel['IMU_ATTI(0):Longitude']  # Columna del Excel con longitud
    altCol = excel['IMU_ATTI(0):relativeHeight:C']  # Columna del Excel con altitud
    timeCol = excel['GPS:Time']  # Columna del Excel con la hora
    timeColDes = np.zeros(len(timeCol))  # Inicializo lista para escrbir hora desfasada

    # Desfasaje de reloj del dron para que coincida con el de la cámara:

    signo = np.sign(desfasaje)
    minutesDes = np.floor(np.abs(desfasaje) / 60)
    secondsDes = np.abs(desfasaje) % 60
    for i in range(len(timeCol)):
        if signo >= 0:
            extramin = np.floor((timeCol[i] % 100 + secondsDes) / 60)
            seconds = (timeCol[i] % 100 + secondsDes) - 60 * extramin

            extrahours = np.floor((np.floor(timeCol[i] / 100) % 100 + extramin + minutesDes) / 60)
            minutes = np.floor(timeCol[i] / 100) % 100 + extramin + minutesDes - 60 * extrahours

            hour = np.floor(timeCol[i] / 10000) + extrahours
        else:
            if timeCol[i] % 100 - secondsDes < 0:
                minrestados = 1
                seconds = 60 + (timeCol[i] % 100 - secondsDes)
            else:
                minrestados = 0
                seconds = timeCol[i] % 100 - secondsDes
            if np.floor(timeCol[i] / 100) % 100 - minrestados - minutesDes < 0:
                hsrestadas = np.floor((np.abs(np.floor(timeCol[i] / 100) % 100 - minrestados - minutesDes)) / 60) + 1
                minutes = 60 - (np.abs(np.floor(timeCol[i] / 100) % 100 - minrestados - minutesDes)) % 60
                if minutes == 60:
                    minutes = 0
                    hsrestadas -= 1
            else:
                hsrestadas = 0
                minutes = np.floor(timeCol[i] / 100) % 100 - minrestados - minutesDes
            hour = np.floor(timeCol[i] / 10000) - hsrestadas
        timeColDes[i] = hour * 10000 + minutes * 100 + seconds

    # Agrega fracciones de segundo:
    
    timeColDes = fracc_sec(timeColDes, tickCol)
    f_yaw = interp1d(timeColDes, yawCol, kind='nearest')
    f_pitch = interp1d(timeColDes, pitchCol, kind='nearest')
    f_roll = interp1d(timeColDes, rollCol, kind='nearest')
    f_lat = interp1d(timeColDes, latCol, kind='nearest')
    f_lon = interp1d(timeColDes, lonCol, kind='nearest')
    f_alt = interp1d(timeColDes, altCol, kind='nearest')

    # Carga fecha y hora desde metadatos y convierte hora en FLOAT (hhmmss.subsec):

    tags = exifread.process_file(binary_img)
    datetime = tags['EXIF DateTimeOriginal']
    subsec = tags['EXIF SubSecTime']
    timeFLOAT = datetimeSTR2timeFLOAT(str(datetime), str(subsec))
    yawDron = f_yaw(timeFLOAT)

    date_str = str(datetime)
    date_num=str2datenum(date_str)
    # Convertir el yaw a un valor entre 0 y 360º en sentido antihorario:
    
    yawDeg=yawDron

    if yawDron < 0:
        yawDron = 360 + yawDron
    yaw = 360 - yawDron
    # Pasa yaw a radianes:
    yaw = math.radians(yaw)

    pitchDron = f_pitch(timeFLOAT)
    pitch = math.radians(pitchDron)
    rollDron = f_roll(timeFLOAT)
    roll = math.radians(rollDron)

    # Lectura de xUTM,yUTM,z desde datos del dron

    latDron = f_lat(timeFLOAT)
    lonDron = f_lon(timeFLOAT)

    z = float(f_alt(timeFLOAT))

    xDron, yDron = LongLat2UTM(lonDron, latDron, 2)

    x = xDron
    y = yDron

    return yaw, pitch, roll, x, y, z,yawDeg,date_num


def camera_dji_features(binary_img):

    """ Lectura de los ángulos de orientación de la cámara y posición (GPS) a partir de los metadatos de una imagen de la camara DJI. Se usan funciones de
        ArmadoMosAuxiliares.py

    Args:
        binary_img: Imagen binaria.

    Returns:
        tuple: Tupla con los valores de los ángulos de orientación yaw, pitch y roll en radianes,
               y las coordenadas x, y, z de la posición de la cámara.
    """
    tags = exifread.process_file(binary_img)
    
    # Yaw, Pitch y Roll en º con de metadatos:

    image=binary_img.read()
    string=str(image)
    
    # Determino las rotaciones del gimbal (del estabilizador), si fuera del vuelo hay que cambiar "Gimbal" por "Flight"
    
    element_pitch="drone-dji:GimbalPitchDegree="
    pitch = string[string.find(element_pitch) + len(element_pitch) : string.find(element_pitch) + len(element_pitch) + 10]
    pitch = np.float64(pitch.split('\"',3)[1])
    
    element_roll="drone-dji:GimbalRollDegree="
    roll = string[string.find(element_roll) + len(element_roll) : string.find(element_roll) + len(element_roll) + 10]
    roll = np.float64(roll.split('\"',3)[1])
    
    element_yaw="drone-dji:GimbalYawDegree="
    yaw = string[string.find(element_yaw) + len(element_yaw) : string.find(element_yaw) + len(element_yaw) + 10]
    yaw = np.float64(yaw.split('\"',3)[1])
    
    element_relative_slt="drone-dji:RelativeAltitude="
    alt = string[string.find(element_relative_slt) + len(element_relative_slt) : string.find(element_relative_slt) + len(element_relative_slt) + 10]
    alt = np.float64(alt.split('\"',3)[1])
    
    pitch = math.radians(pitch+90)
    roll = math.radians(roll)
    
    yawDeg = yaw
    #Convierte yaw a valor entre 0 y 360º en sentido antihorario:
    if yaw < 0:
        yaw = 360 + yaw
    yaw = 360 - yaw
    # Pasa yaw a radianes:
    yaw = math.radians(yaw)

    # Determinacion de la posicion del GPS. Uso de Return Exif tags.

    lat = tags['GPS GPSLatitude']
    long = tags['GPS GPSLongitude']
    date_str=str(tags['Image DateTime'])
    date_num=str2datenum(date_str)
    # print(date_num)
    
    #alt = tags['GPS GPSAltitude']

    aux1=str(long)
    aux2=str(lat)

    a=aux1[1:3]
    if a[-1]==',':
         aux1=aux1[0:1]+'0'+aux1[1:len(aux1)]

    a=aux1[5:7]    
    if a[-1]==',':
         aux1=aux1[0:5]+'0'+aux1[5:len(aux1)]    
    
    a=aux2[1:3]
    if a[-1]==',':
         aux2=aux2[0:1]+'0'+aux2[1:len(aux2)]

    a=aux2[5:7]    
    if a[-1]==',':
         aux2=aux2[0:5]+'0'+aux2[5:len(aux2)]  

    latitude = str2lat_or_long_vec(aux2)
    longitude = str2lat_or_long_vec(aux1)
    
    xCam,yCam=LongLat2UTM(longitude,latitude,1)
    
    x = xCam
    y = yCam

    return yaw, pitch, roll, x, y, alt, yawDeg,date_num

def camera_features(pil_img, binary_img):

    """ Lectura de los ángulos de orientación de la cámara y posición (GPS) a partir de los metadatos de una imagen de la camara multiespectral. Se usan funciones de
        ArmadoMosAuxiliares.py

    Args:
        pil_img: Imagen en formato PIL.
        binary_img: Imagen binaria.

    Returns:
        tuple: Tupla con los valores de los ángulos de orientación yaw, pitch y roll en radianes,
               y las coordenadas x, y, z de la posición de la cámara.
    """

    
    # Leer metadatos de la imagen

    STRwithMetadata = str(pil_img.tag[700])

    # Yaw, Pitch y Roll en º con de metadatos:

    [yaw, pitch, roll] = orientacion(STRwithMetadata)
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    # Convierte yaw a valor entre 0 y 360º en sentido antihorario:
    if yaw < 0:
        yaw = 360 + yaw
    yaw = 360 - yaw
    # Pasa yaw a radianes:
    yawDeg = yaw
    yaw = math.radians(yaw)


    # Determinacion de la posicion del GPS. Uso de Return Exif tags.
    
    tags = exifread.process_file(binary_img)

    lat = tags['GPS GPSLatitude']
    long = tags['GPS GPSLongitude']
    alt = tags['GPS GPSAltitude']

    latitude = str2lat_or_long_vec(str(lat))
    longitude = str2lat_or_long_vec(str(long))
    xCam,yCam=LongLat2UTM(longitude,latitude,1)
    x = xCam
    y = yCam
    z = str2fraction(str(alt))
    
    datetime = tags['EXIF DateTimeOriginal']
    date_str = str(datetime)
    date_num=str2datenum(date_str)

    return yaw, pitch, roll, x, y, z,yawDeg, date_num