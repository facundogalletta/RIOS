from interactive_windows import interactive_window, interactive_window_number,select_path,select_csv,interactive_window_aviso,interactive_window_numbers
from background import background
from prefix import prefijo
import numpy as np
import matplotlib.pyplot as plt
import os

def thermal_processing_setup(img_type, drift, kind_vignetting, kind_slope_accumulation,
                             SourcePath, range_c, band,
                             actual_slope_accumulation,
                             flag_plot_average):
    """
    Configuración del procesamiento térmico (img_type == 2)
    """

    # Valores por defecto (caso NO térmico o sin corrección)
    out = {
        'temp_mean': [],
        'temp_5': [],
        'temp_95': [],
        'coef_corr': [],
        'correction_vignetting': None,
        'DT_vignetting': [],
        'slope': 0,
        'kind_slope_accumulation': '3'
    }

    # --------------------------------
    # Si NO es imagen térmica
    # --------------------------------
    if img_type != 2:
        return out

    # --------------------------------
    # Es imagen térmica
    # --------------------------------
    if drift != '1':
        return out

    # -------- Corrección de bordes --------
    correction_map = {
        '1': 'Original',
        '2': 'Paraboloide',
        '3': 'Gaussian',
        '4': None
    }

    if kind_vignetting not in correction_map:
        interactive_window_aviso('NO SELECCIONÓ UNA OPCIÓN VÁLIDA')
        return out

    correction_vignetting = correction_map[kind_vignetting]
    out['correction_vignetting'] = correction_vignetting

    # -------- Cálculo del promedio de agua fría --------
    if correction_vignetting in ['Paraboloide', 'Gaussian', 'Original']:

        Prefix = 'IMG_'
        path = []

        for k in range(range_c[0], range_c[-1] + 1):
            name = os.path.join(
                SourcePath,
                prefijo(k, Prefix) + str(k) + "_" + str(band) + ".tif"
            )
            path.append(name)

        vignetting = background(path, band)
        DT_vignetting = vignetting - np.min(vignetting)

        out['DT_vignetting'] = DT_vignetting

        if flag_plot_average == '1':
            plt.figure()
            plt.imshow(vignetting, cmap='jet')
            plt.colorbar()
            plt.show()

    # -------- slope --------
    if kind_slope_accumulation == '1':
        slope = actual_slope_accumulation
    elif kind_slope_accumulation == '2':
        slope = interactive_window_number('Enter the new slope')
    elif kind_slope_accumulation == '3':
        slope = 0
    else:
        slope = 0

    out['slope'] = slope
    out['kind_slope_accumulation'] = kind_slope_accumulation

    return out
