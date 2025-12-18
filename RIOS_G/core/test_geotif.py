from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.io import imread
from mpl_toolkits.axes_grid1 import make_axes_locatable

# # Cargar las imágenes originales
# path_r = 'D:/fgalletta/1809-1205/GeoTiff_old/mos_curve_old_0.JPG'
# path_g = 'D:/fgalletta/1809-1205/GeoTiff_old/mos_curve_old_1.JPG'
# path_b = 'D:/fgalletta/1809-1205/GeoTiff_old/mos_curve_old_2.JPG'

# # path_img = 'D:/1809-1205/DJI_0910.JPG'

# imarray_r = np.array(imread(path_r))
# imarray_g = np.array(imread(path_g))
# imarray_b = np.array(imread(path_b))

# # Asegurarse de que las imágenes tengan las mismas dimensiones
# min_height = min(imarray_r.shape[0], imarray_g.shape[0], imarray_b.shape[0])
# min_width = min(imarray_r.shape[1], imarray_g.shape[1], imarray_b.shape[1])

# # Recortar las imágenes al mismo tamaño
# imarray_r = imarray_r[:min_height, :min_width]
# imarray_g = imarray_g[:min_height, :min_width]
# imarray_b = imarray_b[:min_height, :min_width]

# imarray_r = cv2.normalize(imarray_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# imarray_g = cv2.normalize(imarray_g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# imarray_b = cv2.normalize(imarray_b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# imarray=imarray_r-imarray_g
# imarray[imarray>30]=4
# # Concatenar las imágenes en una matriz RGB
# rgb_image = np.dstack((imarray_r, imarray_g, imarray_b))

# # cv2.imwrite('D:/1809-1205/GeoTiff/Merged_GTiff_mosaico_red.JPG', imarray)    

# # Mostrar la imagen resultante
# plt.figure()
# plt.imshow(rgb_image)

# plt.figure()
# plt.imshow(imarray)


# Cargar las imágenes originales
path_r = '../Dragando/Rectificadas/P155/orthorectified_DJI_0279_0.JPG'
path_g = '../Dragando/Rectificadas/P155/orthorectified_DJI_0279_1.JPG'
path_b = '../Dragando/Rectificadas/P155/orthorectified_DJI_0279_2.JPG'

path_or = '../Dragando/DJI_0279.JPG'

# path_eq='../Descargando/Georeferenciadas/P1/Equalizada/p202310271754_DRON.JPG'

# imarray_eq=np.array(imread(path_eq))

# path_img = 'D:/1809-1205/DJI_0910.JPG'

imarray_r = np.array(imread(path_r))
imarray_g = np.array(imread(path_g))
imarray_b = np.array(imread(path_b))

im_or=imread(path_or)
# Asegurarse de que las imágenes tengan las mismas dimensiones
min_height = min(imarray_r.shape[0], imarray_g.shape[0], imarray_b.shape[0])
min_width = min(imarray_r.shape[1], imarray_g.shape[1], imarray_b.shape[1])

# Recortar las imágenes al mismo tamaño
imarray_r = imarray_r[:min_height, :min_width]
imarray_g = imarray_g[:min_height, :min_width]
imarray_b = imarray_b[:min_height, :min_width]

# imarray_r = cv2.normalize(imarray_r, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# imarray_g = cv2.normalize(imarray_g, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# imarray_b = cv2.normalize(imarray_b, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# imarray_p = ((imarray_r + imarray_g) / 2 - imarray_b).clip(0, 255).astype(np.uint8)

# imarray_d=imarray_b/(imarray_r-imarray_g)

# imarray_rb=imarray_r-imarray_b


# imarray[imarray>30]=4
# # Concatenar las imágenes en una matriz RGB

rgb_image = np.dstack((imarray_r, imarray_g, imarray_b))

# # # cv2.imwrite('D:/1809-1205/GeoTiff/Merged_GTiff_mosaico_red.JPG', imarray)    

# # # Mostrar la imagen resultante

# blurred_image = cv2.GaussianBlur(imarray_p, (55, 55), 0)
# median_filtered_image = cv2.medianBlur(imarray_p, 65)

# # Aplicar el filtro de dilatación
# kernel_size = 1  # Puedes ajustar el tamaño del kernel según sea necesario
# kernel = np.ones((kernel_size, kernel_size), np.uint8)
# imarray_p_dilatada = cv2.dilate(imarray_p, kernel, iterations=10)
# edges=cv2.Canny(imarray_p_dilatada,0,6,100,L2gradient = True)


# edges=cv2.Canny(imarray_p,0,10,100,L2gradient = True)



#agregar imagen rgb y original en subplots
fig,ax=plt.subplots(1,2)
ax[1].imshow(rgb_image)
ax[0].imshow(im_or)

# Guardar la imagen rectificada rgb
# inveritr canales para usar open cv
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
cv2.imwrite('../Dragando/Rectificadas/P155/rgb_orthorectified_DJI_0279.JPG', rgb_image)

# path='D:/fgalletta/timag/Source Path/Salida/GeoTiff/Merged_GTiff_correccion_thermal_no_bordes_original.tif'
# # Cargar la imagen original
# imarray = np.array(imread(path))
# plt.figure()
# plt.imshow(imarray,vmin=21,vmax=28,cmap='jet')
# plt.colorbar()

# path='D:/fgalletta/timag/Source Path/Salida/GeoTiff/Mos.tif'
# # Cargar la imagen original
# imarray = np.array(imread(path))
# plt.figure()
# plt.imshow(imarray,vmin=18,vmax=24,cmap='jet')
# plt.colorbar()

plt.show()