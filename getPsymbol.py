def getPsymbol(P12, P13=None, P23=None, alpha12=0.05, alpha13=0.05, alpha23=0.05):
    sig12 = ' '
    sig13 = ' '
    sig23 = ' '

    if P13 is not None:
        if P12 < alpha12:
            sig12 = '*'
    
    if P23 is not None:
        if P13 < alpha13:
            sig13 = '†'
    
    if P23 is not None:
        if P23 < alpha23:
            sig23 = '§'
    
    return sig12, sig13, sig23

