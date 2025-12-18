def orientacion(STRwithMetadata):
    """
    Function that reads the yaw, pitch and roll angles of the metadata image
    
    input:
        STRwithMetadata: string with the metadata of the image
    
    output:
        YAW_in_Deg: yaw angle in degrees
        PITCH_in_Deg: pitch angle in degrees
        ROLL_in_Deg: roll angle in degrees
    """
    
    
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