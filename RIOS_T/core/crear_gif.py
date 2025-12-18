from PIL import Image
import numpy as np
import glob

SourcePath="D:/fgalletta/OneDrive - Facultad de Ingeniería/01_Investigación/04_Publicaciones/Paper Dron/03-Avances/Termica/Figuras/Vuelo/"
DstPath = "D:/fgalletta/OneDrive - Facultad de Ingeniería/01_Investigación/04_Publicaciones/Paper Dron/03-Avances/Termica/Figuras/Vuelo/gif/"


# Lista de nombres de archivo de las imágenes que deseas agregar al GIF
image_file_names = glob.glob(SourcePath+'*.png')

# image_file_names2 = [DstPath+'intersects_'+str(i)+'.png' for i in range(-60,0,10)]

# image_files_names3=image_file_names2[::-1]

# image_file_names=image_file_names1+image_file_names2[1:]+image_files_names3

# Lista de duraciones de visualización de cada imagen en milisegundos (2000 ms = 2 segundos)
durations = [100]*len(image_file_names)

# Abre las imágenes y las agrega a una lista
images = [Image.open(filename) for filename in image_file_names]
# print(images)
# Guarda el GIF animado
output_gif = DstPath+'pluma_rapido.gif'
images[0].save(
output_gif,
    save_all=True,
    append_images=images[1:],
    duration=durations,
    loop=0  # Establece loop=0 para que el GIF se reproduzca infinitamente
)

print(f'GIF animado guardado en "{output_gif}"')
