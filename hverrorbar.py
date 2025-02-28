import matplotlib.pyplot as plt
import numpy as np

def hverrorbar(x, y, lux, luy, **kwargs):
    """
    ERRORBAR Error bar plot.
    ERRORBAR(x, y, lux, luy, **kwargs) plots the graph of vector x vs. vector y with
    error bars specified by the vectors L and U. L and U contain the
    lower and upper error ranges for each point in Y. Each error bar
    is L(i) + U(i) long and is drawn a distance of U(i) above and L(i)
    below the points in (X,Y). The vectors x, y, L and U must all be
    the same length. If x, y, L and U are matrices then each column
    produces a separate line.
    
    H = hverrorbar(...) returns a list of Line2D objects.
    """

    if isinstance(x, np.ndarray):  # Verifica si x es un array
        npt = len(x)
    else:  # Maneja el caso escalar
        npt = 1
        
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    lx = lux[0, :]
    ux = lux[1, :]
    ly = luy[0, :]
    uy = luy[1, :]

    ux = np.abs(ux)
    lx = np.abs(lx)
    uy = np.abs(uy)
    ly = np.abs(ly)

    teex = (max(y) - min(y)) / 100  # make tee .02 x-distance for error bars
    if teex == 0:
        teex = max([max(ly), max(uy)]) / 4

    teey = (max(x) - min(x)) / 100  # make tee .02 x-distance for error bars
    if teey == 0:
        teey = max([max(lx), max(ux)]) / 4

    xly = x - teey
    xry = x + teey
    ytopy = y + uy
    yboty = y - ly

    xlx = x - lx
    xrx = x + ux
    ytopx = y + teex
    ybotx = y - teex
    n = y.shape[0]

    # build up nan-separated vector for bars
    xby = np.zeros(npt * 9)
    xby[0::9] = x
    xby[1::9] = x
    xby[2::9] = np.nan
    xby[3::9] = xly
    xby[4::9] = xry
    xby[5::9] = np.nan
    xby[6::9] = xly
    xby[7::9] = xry
    xby[8::9] = np.nan

    yby = np.zeros(npt * 9)
    yby[0::9] = ytopy
    yby[1::9] = yboty
    yby[2::9] = np.nan
    yby[3::9] = ytopy
    yby[4::9] = ytopy
    yby[5::9] = np.nan
    yby[6::9] = yboty
    yby[7::9] = yboty
    yby[8::9] = np.nan

    xbx = np.zeros(npt * 9)
    xbx[0::9] = xlx
    xbx[1::9] = xlx
    xbx[2::9] = np.nan
    xbx[3::9] = xlx
    xbx[4::9] = xrx
    xbx[5::9] = np.nan
    xbx[6::9] = xrx
    xbx[7::9] = xrx
    xbx[8::9] = np.nan

    ybx = np.zeros(npt * 9)
    ybx[0::9] = ytopx
    ybx[1::9] = ybotx
    ybx[2::9] = np.nan
    ybx[3::9] = y
    ybx[4::9] = y
    ybx[5::9] = np.nan
    ybx[6::9] = ytopx
    ybx[7::9] = ybotx
    ybx[8::9] = np.nan

    # Plot graph and bars
    h = plt.plot(xby, yby, **kwargs)
    h += plt.plot(xbx, ybx, **kwargs)

    return h


#plt.figure()
# Ejemplo de uso
#x = np.arange(1, 11)
#y = np.sin(x)


#xx=[np.mean(x)]
#yy=[np.mean(y)]
   
#lux = np.std(x) * np.array([[1], [1]])
#luy=np.std(y) * np.array([[1], [1]])
#hverrorbar(xx, yy, lux, luy, color='k', linestyle='-', linewidth=1)