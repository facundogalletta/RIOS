def datetimeSTR2timeFLOAT(datetime, subsec):
    """
    Function to convert datetime string to float
    
    input:
        datetime: datetime string
        subsec: subseconds  
    
    return: 
        timeFLOAT: float
    """
    
    hour = int(datetime[11:13])
    minute = int(datetime[14:16])
    seconds = int(datetime[17:19])
    timeFLOAT = float(hour * 10000 + minute * 100 + seconds)
    subsecFLOAT = float('0.' + subsec)
    timeFLOAT = timeFLOAT + subsecFLOAT
    return timeFLOAT