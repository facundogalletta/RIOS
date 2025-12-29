def build_output_paths(SourcePath, type_process, band, ini, fin,
                       drift=None, kind_vignetting=None, kind_slope_accumulation=None,correction_drift_method=None,intersection_values=None):
    """
    Bulid out folder Geo y Rect for each processing.
    """

    if type_process == '3':
        DstPathGeo  = f"{SourcePath}/Geo_Ref_{band}_{ini}_{fin}"
        DstPathRect = f"{SourcePath}/Rec_Ref_{band}_{ini}_{fin}"

    elif type_process == '2':
        DstPathGeo  = f"{SourcePath}/Geo_Term"
        DstPathRect = f"{SourcePath}/Rec_Term"

        if drift == '1':
            
            DstPathGeo  += "_D"
            DstPathRect += "_D"
            
            if correction_drift_method == 'mult':
                DstPathGeo  += "Mult"
                DstPathRect += "Mult"
            else:
                DstPathGeo  += "Add"
                DstPathRect += "Add"
                
            if intersection_values == 'obl':
                DstPathGeo  += "Obl"
                DstPathRect += "Obl"
            else:
                DstPathGeo  += "Rec"
                DstPathRect += "Rec"

            kind_vignetting_map = {
                '1': '_A',
                '2': '_P',
                '3': '_G',
                '4': '_NE'
            }
            if kind_vignetting in kind_vignetting_map:
                DstPathGeo  += kind_vignetting_map[kind_vignetting]
                DstPathRect += kind_vignetting_map[kind_vignetting]

            if kind_slope_accumulation in ['1', '2']:
                suf = '_S'
            else:
                suf = '_NS'

            DstPathGeo  += suf
            DstPathRect += suf

        else:
            DstPathGeo  += "_RAW"
            DstPathRect += "_RAW"

        DstPathGeo  += f"_{ini}_{fin}"
        DstPathRect += f"_{ini}_{fin}"

    elif type_process == '1':
        DstPathGeo  = f"{SourcePath}/Geo_{band}_{ini}_{fin}"
        DstPathRect = f"{SourcePath}/Rec_{band}_{ini}_{fin}"

    else:
        raise ValueError("type_process non valid")

    return DstPathGeo, DstPathRect
