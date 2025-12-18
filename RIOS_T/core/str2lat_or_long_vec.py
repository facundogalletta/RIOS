from core.str2fraction import str2fraction
def str2lat_or_long_vec(string):
    """
    Convert text string latitude or longitude to vector string
    
    input:
        string: text string
        
    return: 
        coord: vector string
    """
    
    deg = string[1:3]
    minutes = string[5:7]
    sec = string[9:len(string) - 1]

    coord = [float(deg), float(minutes), str2fraction(sec)]

    return coord