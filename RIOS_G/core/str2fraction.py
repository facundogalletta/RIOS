def str2fraction(string):
    """
    Convert text string fracction to float
    
    input:
        string: text string
    
    return: 
        fraction: float
    """
    
    for i in range(len(string)):
        if string[i] == '/':
            indDiv = i
    try:  # Significa que string vino como un número entero
        num = string[0:indDiv]
        den = string[indDiv + 1:len(string)]
        num = float(num)
        den = float(den)

        fraction = num / den
    except:
        fraction = float(string)

    return fraction