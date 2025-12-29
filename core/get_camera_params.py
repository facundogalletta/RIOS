from interactive_windows import interactive_window, interactive_window_number,select_path, select_csv, interactive_window_aviso,interactive_window_numbers

def get_camera_params(type_process, cam, band=None):
    """
    Devuelve parámetros de cámara según tipo de proceso y cámara.
    """

    params = {}

    # ---------------------------
    # PROCESO 1: RGB / Multiespectral
    # ---------------------------
    if type_process == '1':

        if cam == '1':  # Micasense Altum
            params.update(dict(
                width=2015,
                height=1496,
                FOVwidth=48,
                FOVheight=37,
                prefix="IMG_",
                img_type=1
            ))

        elif cam == '2':  # Thermal (no válida)
            raise ValueError("El sensor térmico no corresponde a este procesamiento")

        elif cam == '3':  # RGB
            params.update(dict(
                width=5280,
                height=2970,
                FOVwidth=64.6869828967688,
                FOVheight=39.21135483264528,
                prefix="DJI_",
                band=0,
                img_type=1
            ))

        elif cam == '4':
            params.update(dict(
                width=4000,
                height=2250,
                FOVwidth=71.19928287983431,
                FOVheight=43.86979719641275,
                prefix="DJI_",
                band=0,
                img_type=1
            ))

        elif cam == '5':
            params.update(dict(
                width=4000,
                height=2250,
                FOVwidth=64.6869828967688,
                FOVheight=39.21135483264528,
                prefix="DJI_",
                band=0,
                img_type=1
            ))

        elif cam == '6':  # Cámara definida por el usuario
            params['width'] = int(interactive_window_number('Width (px)'))
            params['height'] = int(interactive_window_number('Height (px)'))
            params['FOVwidth'] = int(interactive_window_number(
                'Field of view in the long side (width)'))
            params['FOVheight'] = int(interactive_window_number(
                'Field of view in the short side (height)'))
            params['prefix'] = interactive_window(
                ['Ejemplos', 'DJI_', 'IMG_'],
                'Definir el prefijo de las imagenes antes de la numeración'
            )
            params['img_type'] = 1

        else:
            raise ValueError("Cámara no válida")

    # ---------------------------
    # PROCESO 2: Térmica
    # ---------------------------
    elif type_process == '2':

        if cam == '2':
            params.update(dict(
                width=160,
                height=120,
                FOVwidth=57,
                FOVheight=44,
                prefix="IMG_",
                band=6,
                img_type=2
            ))
        else:
            raise ValueError("Cámara no válida para proceso térmico")

    # ---------------------------
    # PROCESO 3: Reflectancia
    # ---------------------------
    elif type_process == '3':

        if cam == '1':
            params.update(dict(
                width=2015,
                height=1496,
                FOVwidth=48,
                FOVheight=37,
                prefix="IMG_",
                img_type=3
            ))

        elif cam == '3':
            params.update(dict(
                width=5280,
                height=2970,
                FOVwidth=64.6869828967688,
                FOVheight=39.21135483264528,
                prefix="DJI_",
                band=0,
                img_type=3
            ))

        elif cam in ['4', '5']:
            params.update(dict(
                width=4000,
                height=2250,
                FOVwidth=64.6869828967688,
                FOVheight=39.21135483264528,
                prefix="DJI_",
                band=0,
                img_type=3
            ))

        elif cam == '6':
            params['width'] = int(interactive_window_number('Width (px)'))
            params['height'] = int(interactive_window_number('Height (px)'))
            params['FOVwidth'] = int(interactive_window_number(
                'Field of view in the long side (width)'))
            params['FOVheight'] = int(interactive_window_number(
                'Field of view in the short side (height)'))
            params['prefix'] = interactive_window(
                ['Ejemplos', 'DJI_', 'IMG_'],
                'Definir el prefijo de las imagenes antes de la numeración'
            )
            params['img_type'] = 3

        else:
            raise ValueError("Cámara no válida")

    else:
        raise ValueError("type_process no válido")

    return params
