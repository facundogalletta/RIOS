
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt

import core.micasense.capture as capture
import core.micasense.imageutils as imageutils
from core.fit_promedio import fit_paraboloide,fit_gaussian

def rad_sky_ref(sky_names):
    """ Funcion para obtener el promedio de radiancias de la imagen de referencia del cielo
    
    Args:
        sky_names (array): Arreglo con los nombres de las imagenes de cielo.
        
    Returns:
        vec_rad_sky_means (array): Arreglo con las radiancias promedio de las imagenes de cielo.
    """        
    if sky_names is not None:
        capture_skyref = capture.Capture.from_filelist(sky_names)
        vec_rad_sky_means = []
        for i in range(5):
            a = capture_skyref.images[i].radiance()
            M, N = a.shape
            rad_mean = a[M // 3:M - M // 3, N // 3:N - N // 3].mean()
            vec_rad_sky_means.append(rad_mean)
    else:
        vec_rad_sky_means = [0, 0, 0, 0, 0]
    return vec_rad_sky_means
    
def aligned_capture(capture, warp_matrices, warp_mode, cropped_dimensions, or_size, img_type = 'reflectance',interpolation_mode=cv2.INTER_LANCZOS4):
    """ Funcion para alinear con demas lentes.

    Args:
        capture_img (Imagen cargada de micasense.capture): Imagen a la cual se le aplica correcion de distorsion
        warp_matrices (array): Imagen de reordenamiento de pixeles para alinear.
        cropped_dimensions (tuble): Dimensiones del recorte a aplicar a la sin distorsion para descartar bordes. Recorte en base a dimension or_size
        or_size (tuple) : Dimensiones de salida de la imagen sin distorsion y alineada, aun con bordes a descartar.
        img_type (string): Tipo de imagen a la cual se aplica correccion. 'reflectance' o 'radiance'.
        
    Returns:
        img_cropped (array) : Imagen sin distorsion y alineada con demas bandas. 
    """
    
    width, height = or_size
    
    if len(capture.images) == 1:
        im_aligned = np.zeros((height,width,1), dtype=np.float32 )
        
        if img_type == 'reflectance':
            img = capture.images[0].undistorted_reflectance()
        else:
            img = capture.images[0].undistorted_radiance()
            # plt.figure(figsize=(12,5))
            # plt.imshow(img, cmap='jet',vmax=21.6,vmin=20.2)
            # plt.axis('off')
            # plt.colorbar()
            # plt.tight_layout()
            # plt.savefig('./Figs/img_undist'+'.png', dpi=500, bbox_inches='tight')
        
        if warp_mode != cv2.MOTION_HOMOGRAPHY:
            im_aligned[:,:,0] = cv2.warpAffine(img,
                                            warp_matrices[0],
                                            (width,height),
                                            flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
            # plt.figure(figsize=(12,5))
            # plt.imshow(cv.resize(im_aligned, (width, height), cv.INTER_CUBIC), cmap='jet',vmax=21.6,vmin=20.2)
            # plt.axis('off')
            # plt.colorbar()
            # plt.tight_layout()
            # plt.savefig('./Figs/img_align'+'.png', dpi=500, bbox_inches='tight')
        else:
            im_aligned[:,:,0] = cv2.warpPerspective(img,
                                                warp_matrices[0],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
            # plt.figure(figsize=(12,5))
            # plt.imshow(cv.resize(im_aligned, (width, height), cv.INTER_CUBIC), cmap='jet',vmax=21.6,vmin=20.2)
            # plt.axis('off')
            # plt.colorbar()
            # plt.tight_layout()
            # plt.savefig('./Figs/img_align'+'.png', dpi=500, bbox_inches='tight')
            
        (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
        im_cropped = im_aligned[top:top+h, left:left+w][:]
        # plt.figure(figsize=(12,5))
        # plt.imshow(cv.resize(im_cropped, (width, height), cv.INTER_CUBIC), cmap='jet',vmax=21.6,vmin=20.2)
        # plt.axis('off')
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('./Figs/img_recorte'+'.png', dpi=500, bbox_inches='tight')

    else:
        im_aligned = np.zeros((height,width,len(warp_matrices)), dtype=np.float32 )

        for i in range(len(warp_matrices)):
            
            if img_type == 'reflectance':
                img = capture.images[i].undistorted_reflectance()
            else:
                img = capture.images[i].undistorted_radiance()
            

            if warp_mode != cv2.MOTION_HOMOGRAPHY:
                im_aligned[:,:,i] = cv2.warpAffine(img,
                                                warp_matrices[str(i+1)],
                                                (width,height),
                                                flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
            else:
                im_aligned[:,:,i] = cv2.warpPerspective(img,
                                                    warp_matrices[str(i+1)],
                                                    (width,height),
                                                    flags=interpolation_mode + cv2.WARP_INVERSE_MAP)
        (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
        im_cropped = im_aligned[top:top+h, left:left+w][:]

    return im_cropped

def aligned_capture_raw(pil_img, band,warp_matrices, warp_mode, cropped_dimensions, or_size,interpolation_mode=cv2.INTER_LANCZOS4):
    """ Funcion para alinear con demas lentes.

    Args:
        capture_img (Imagen cargada de micasense.capture): Imagen a la cual se le aplica correcion de distorsion
        warp_matrices (array): Imagen de reordenamiento de pixeles para alinear.
        cropped_dimensions (tuble): Dimensiones del recorte a aplicar a la sin distorsion para descartar bordes. Recorte en base a dimension or_size
        or_size (tuple) : Dimensiones de salida de la imagen sin distorsion y alineada, aun con bordes a descartar.
        img_type (string): Tipo de imagen a la cual se aplica correccion. 'reflectance' o 'radiance'.
        
    Returns:
        img_cropped (array) : Imagen sin distorsion y alineada con demas bandas. 
    """
    
    
    width, height = or_size
    img = np.array(pil_img, dtype=float)  # lo paso a float para normalizar

    # normalización min-max a rango 0–255
    img = (img - img.min()) / (img.max() - img.min()) * 255
    img = img.astype(np.uint8)  # ahora en 0–255
    
    # Eliminar saturacion
    
    im_aligned = np.zeros((height,width,1), dtype=np.float32 )
    im_aligned[:,:,0] =  cv2.warpPerspective(img,warp_matrices[str(band)],(width,height), flags=interpolation_mode + cv2.WARP_INVERSE_MAP)

    (left, top, w, h) = tuple(int(i) for i in cropped_dimensions)
    im_cropped = im_aligned[top:top+h, left:left+w][:]

    return im_cropped

def calibration_reflectance(capture_img, band, height, width,panel_names=None,sky_names=None):

    """ Funcion para alinear con demas lentes y correccion de distorsion y de efectos de bordes (opcional) de imagenes. 
        La distorsion se corrige desde implementacion GitHub de micasense.

    Args:
        capture_img (Imagen cargada de micasense.capture): Imagen a la cual se le aplica correcion de distorsion
        band (int) : Numero de banda espectarl a la que se le aplica la alineacion y correccoines.
        height (int) : Altura de la imagen original a corregir
        width (int) : Ancho de la imagen original a corregir
        DT_promedio_agua_fria (array) : Imagen promedio de variacion de temperatura en agua fria el dia del vuelo. Se asume que variaciones son efectos de borde.
        correction_bordes (string) : Especificacion de como corregir los efectos de bordes (en caso que se corrijan).
        panel_names (array) : Arreglo con los nombres de las imagenes de panel.
        sky_names (array) : Arreglo con los nombres de las imagenes de cielo.  

    Returns:
        img (array) : Imagen "calibrada" alineada, sin distorsion y sin efectos de bordes (en caso de estar definidos).
    """

    # Definir matrices de reordenamiento para alinear lentes.

    warp_matrices = {
	"1": np.array([[ 1.0000991e+00, -4.9471273e-03, -2.6685207e+01],
			[ 3.3375097e-03,  1.0002422e+00, -1.9665100e+01],
			[-3.8116281e-07, -1.1621883e-06,  1.0000000e+00]]),
	"2": np.array([[1., 0., 0.],
			 [0., 1., 0.],
			 [0., 0., 1.]]),
	"3": np.array([[ 9.99950647e-01, -3.60592664e-03, -1.01690235e+01],
			 [ 2.57935910e-03,  9.99263883e-01, -1.61942997e+01],
			 [ 3.36570992e-07, -1.28640738e-06,  1.00000000e+00]]),
	"4": np.array([[ 1.0026586e+00, -3.4015453e-03, -2.9431755e+01],
			 [ 3.1610006e-03,  1.0026866e+00, -1.0079272e+01],
			 [ 7.0966064e-07, -6.5917897e-07,  1.0000000e+00]]),
	"5": np.array([[ 1.0012406e+00 , 7.0302095e-04, -1.0447843e+01],
			 [-1.7317060e-03,  9.9995524e-01,  7.6750369e+00],
			 [ 7.2561721e-07, -1.5492838e-06,  1.0000000e+00]]),
	"6": np.array([[ 6.16236288e-02, -1.06880448e-03,  1.55822672e+01],
			 [ 9.17530878e-04,  6.20836284e-02,  8.75423824e+00],
			 [-2.50129098e-06, -2.38021784e-07,  1.00000000e+00]])
    }

    # Seleccionar matriz del lente a alinear
    # warp_matrices=[warp_matrices[str(band)]]
    
    vec_rad_sky_means = rad_sky_ref(sky_names)
    panelCap = capture.Capture.from_filelist(panel_names)

    # Initialize raw and reflectance images for all bands
    if panelCap is not None:
        if panelCap.panel_albedo() is not None:
            panel_reflectance_by_band = panelCap.panel_albedo()
        else:
            panel_reflectance_by_band = [0.67, 0.69, 0.68, 0.61, 0.67]  # RedEdge band_index order
        panel_irradiance = panelCap.panel_irradiance(panel_reflectance_by_band)
        capture_img.undistorted_reflectance(panel_irradiance, vec_rad_sky_means)
    else:
        capture_img.undistorted_reflectance(capture_img.dls_irradiance(), vec_rad_sky_means)

    # Dimensiones de recorte luego de corregir de distorsion, se descartan bordes.
    # Este recorte esta definido para las dimensiones de la imagen sin distorsion (2064,1544).

    cropped_dimensions = (40.0, 26.0, 2015.0, 1496.0)

    if band == 6:
        img_type = 'radiance'
    else:
        img_type = 'reflectance'
    
    im_aligned = aligned_capture(capture_img, warp_matrices, cv2.MOTION_HOMOGRAPHY, cropped_dimensions, (2064, 1544),
                                            img_type=img_type)

    img = im_aligned[:, :, band-1]
    
    #Mantener la resolucion de la imagen horiginal 

    img = cv.resize(img, (width, height), cv.INTER_CUBIC)


    return img

def calibration_raw(pil_img, band, height, width,panel_names=None,sky_names=None):

    """ Funcion para alinear con demas lentes y correccion de distorsion y de efectos de bordes (opcional) de imagenes. 
        La distorsion se corrige desde implementacion GitHub de micasense.

    Args:
        capture_img (Imagen cargada de micasense.capture): Imagen a la cual se le aplica correcion de distorsion
        band (int) : Numero de banda espectarl a la que se le aplica la alineacion y correccoines.
        height (int) : Altura de la imagen original a corregir
        width (int) : Ancho de la imagen original a corregir
        DT_promedio_agua_fria (array) : Imagen promedio de variacion de temperatura en agua fria el dia del vuelo. Se asume que variaciones son efectos de borde.
        correction_bordes (string) : Especificacion de como corregir los efectos de bordes (en caso que se corrijan).
        panel_names (array) : Arreglo con los nombres de las imagenes de panel.
        sky_names (array) : Arreglo con los nombres de las imagenes de cielo.  

    Returns:
        img (array) : Imagen "calibrada" alineada, sin distorsion y sin efectos de bordes (en caso de estar definidos).
    """

    # Definir matrices de reordenamiento para alinear lentes.

    warp_matrices = {
	"1": np.array([[ 1.0000991e+00, -4.9471273e-03, -2.6685207e+01],
			[ 3.3375097e-03,  1.0002422e+00, -1.9665100e+01],
			[-3.8116281e-07, -1.1621883e-06,  1.0000000e+00]]),
	"2": np.array([[1., 0., 0.],
			 [0., 1., 0.],
			 [0., 0., 1.]]),
	"3": np.array([[ 9.99950647e-01, -3.60592664e-03, -1.01690235e+01],
			 [ 2.57935910e-03,  9.99263883e-01, -1.61942997e+01],
			 [ 3.36570992e-07, -1.28640738e-06,  1.00000000e+00]]),
	"4": np.array([[ 1.0026586e+00, -3.4015453e-03, -2.9431755e+01],
			 [ 3.1610006e-03,  1.0026866e+00, -1.0079272e+01],
			 [ 7.0966064e-07, -6.5917897e-07,  1.0000000e+00]]),
	"5": np.array([[ 1.0012406e+00 , 7.0302095e-04, -1.0447843e+01],
			 [-1.7317060e-03,  9.9995524e-01,  7.6750369e+00],
			 [ 7.2561721e-07, -1.5492838e-06,  1.0000000e+00]]),
	"6": np.array([[ 6.16236288e-02, -1.06880448e-03,  1.55822672e+01],
			 [ 9.17530878e-04,  6.20836284e-02,  8.75423824e+00],
			 [-2.50129098e-06, -2.38021784e-07,  1.00000000e+00]])
    }

    # warp_matrices=[warp_matrices[str(band)]]
    # Este recorte esta definido para las dimensiones de la imagen sin distorsion (2064,1544).

    cropped_dimensions = (40.0, 26.0, 2015.0, 1496.0)
    
    im_aligned = aligned_capture_raw(pil_img,band, warp_matrices, cv2.MOTION_HOMOGRAPHY, cropped_dimensions, (width, height))
    
    img = im_aligned
    #Mantener la resolucion de la imagen horiginal 

    img = cv.resize(img, (width, height), cv.INTER_CUBIC)

    # Quitar efectos de bordes, en caso que se definan.
    return img

def calibration_thermal(capture_img, band, height, width,DT_promedio_agua_fria,correction_bordes,panel_names=None,sky_names=None):

    """ Funcion para alinear con demas lentes y correccion de distorsion y de efectos de bordes (opcional) de imagenes. 
        La distorsion se corrige desde implementacion GitHub de micasense.

    Args:
        capture_img (Imagen cargada de micasense.capture): Imagen a la cual se le aplica correcion de distorsion
        band (int) : Numero de banda espectarl a la que se le aplica la alineacion y correccoines.
        height (int) : Altura de la imagen original a corregir
        width (int) : Ancho de la imagen original a corregir
        DT_promedio_agua_fria (array) : Imagen promedio de variacion de temperatura en agua fria el dia del vuelo. Se asume que variaciones son efectos de borde.
        correction_bordes (string) : Especificacion de como corregir los efectos de bordes (en caso que se corrijan).
        panel_names (array) : Arreglo con los nombres de las imagenes de panel.
        sky_names (array) : Arreglo con los nombres de las imagenes de cielo.  

    Returns:
        img (array) : Imagen "calibrada" alineada, sin distorsion y sin efectos de bordes (en caso de estar definidos).
    """

    # Definir matrices de reordenamiento para alinear lentes.

    warp_matrices = {
	"1": np.array([[ 1.0000991e+00, -4.9471273e-03, -2.6685207e+01],
			[ 3.3375097e-03,  1.0002422e+00, -1.9665100e+01],
			[-3.8116281e-07, -1.1621883e-06,  1.0000000e+00]]),
	"2": np.array([[1., 0., 0.],
			 [0., 1., 0.],
			 [0., 0., 1.]]),
	"3": np.array([[ 9.99950647e-01, -3.60592664e-03, -1.01690235e+01],
			 [ 2.57935910e-03,  9.99263883e-01, -1.61942997e+01],
			 [ 3.36570992e-07, -1.28640738e-06,  1.00000000e+00]]),
	"4": np.array([[ 1.0026586e+00, -3.4015453e-03, -2.9431755e+01],
			 [ 3.1610006e-03,  1.0026866e+00, -1.0079272e+01],
			 [ 7.0966064e-07, -6.5917897e-07,  1.0000000e+00]]),
	"5": np.array([[ 1.0012406e+00 , 7.0302095e-04, -1.0447843e+01],
			 [-1.7317060e-03,  9.9995524e-01,  7.6750369e+00],
			 [ 7.2561721e-07, -1.5492838e-06,  1.0000000e+00]]),
	"6": np.array([[ 6.16236288e-02, -1.06880448e-03,  1.55822672e+01],
			 [ 9.17530878e-04,  6.20836284e-02,  8.75423824e+00],
			 [-2.50129098e-06, -2.38021784e-07,  1.00000000e+00]])
    }

    if len(capture_img.images)==1:
        warp_matrices=[warp_matrices[str(band)]]

    # Dimensiones de recorte luego de corregir de distorsion, se descartan bordes.
    # Este recorte esta definido para las dimensiones de la imagen sin distorsion (2064,1544).

    cropped_dimensions = (40.0, 26.0, 2015.0, 1496.0)
    
    img_type = 'radiance'
    
    im_aligned = aligned_capture(capture_img, warp_matrices, cv2.MOTION_HOMOGRAPHY, cropped_dimensions, (2064, 1544),
                                            img_type=img_type)

    if len(capture_img.images)==1:
        img = im_aligned[:, :, 0]  
    else:
        img = im_aligned[:, :, band-1]
    
    #Mantener la resolucion de la imagen horiginal 

    img = cv.resize(img, (width, height), cv.INTER_CUBIC)

    # Quitar efectos de bordes, en caso que se definan.
    
    if band == 6:
        if correction_bordes is not None:
            if correction_bordes=='Original':
                img=img-DT_promedio_agua_fria
            elif correction_bordes=='Paraboloide':
                DT=fit_paraboloide(DT_promedio_agua_fria)
                img=img-DT
            elif correction_bordes=='Gaussian':
                DT=fit_gaussian(DT_promedio_agua_fria)
                img=img-DT

    return img