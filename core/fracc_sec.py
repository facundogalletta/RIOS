def fracc_sec(timeColDes, tickCol):
    """"
    
    
        
    """
    sec_actual = timeColDes[0]
    timeColDes[0] = timeColDes[0] + 0.05
    for i in range(1, len(timeColDes)):
        if timeColDes[i] == sec_actual:
            timeColDes[i] = timeColDes[i - 1] + (tickCol[i] - tickCol[i - 1]) / 4500000
        else:
            sec_actual = timeColDes[i]
            timeColDes[i] = timeColDes[i] + 0.05
    return timeColDes
