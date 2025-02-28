import numpy as np
import matplotlib.pyplot as plt

def hverrorbar50(x, y, lux, luy, *varargin):
    """
    Error bar plot.

    Plots the graph of vector x vs. vector y with error bars specified by the vectors lux and luy.
    lux and luy contain the lower and upper error ranges for each point in y.
    Each error bar is lux(i) + luy(i) long and is drawn a distance of luy(i) above and lux(i) below the points in (x, y).
    The vectors x, y, lux, and luy must all be the same length.
    If x, y, lux, and luy are matrices, then each column produces a separate line.
    
    Example:
        x = np.arange(1, 11)
        y = np.sin(x)
        e = np.std(y) * np.ones_like(x)
        hverrorbar(x, y, e, e)
    """
    # Checking which plotting API should be used
    nVarargs = len(varargin)
    
    npt = len(x)
    x = np.reshape(x, (-1,))
    y = np.reshape(y, (-1,))
    
    lx = lux[0, :]
    ux = lux[1, :]
    ly = luy[0, :]
    uy = luy[1, :]
    
    ux = np.abs(ux)
    lx = np.abs(lx)
    uy = np.abs(uy)
    ly = np.abs(ly)
    
    teex = (np.max(y) - np.min(y)) / 100
    if teex == 0:
        teex = np.max([ly, uy]) / 4
    
    teey = (np.max(x) - np.min(x)) / 100
    if teey == 0:
        teey = np.max([lx, ux]) / 4
    
    xly = x - teey
    xry = x + teey
    ytopy = y + uy
    yboty = y - ly
    
    xlx = x - lx
    xrx = x + ux
    ytopx = y + teex
    ybotx = y - teex
    n = y.shape[1] if y.ndim > 1 else 1
    
    # Build up nan-separated vectors for bars
    xby = np.zeros((npt * 9, n))
    xby[0:9:n, :] = x[:, np.newaxis]
    xby[1:9:n, :] = x[:, np.newaxis]
    xby[2:9:n, :] = np.nan
    xby[3:9:n, :] = xly[:, np.newaxis]
    xby[4:9:n, :] = xry[:, np.newaxis]
    xby[5:9:n, :] = np.nan
    xby[6:9:n, :] = xly[:, np.newaxis]
    xby[7:9:n, :] = xry[:, np.newaxis]
    xby[8:9:n, :] = np.nan
    
    yby = np.zeros((npt * 9, n))
    yby[0:9:n, :] = ytopy[:, np.newaxis]
    yby[1:9:n, :] = yboty[:, np.newaxis]
    yby[2:9:n, :] = np.nan
    yby[3:9:n, :] = ytopy[:, np.newaxis]
    yby[4:9:n, :] = ytopy[:, np.newaxis]
    yby[5:9:n, :] = np.nan
    yby[6:9:n, :] = yboty[:, np.newaxis]
    yby[7:9:n, :] = yboty[:, np.newaxis]
    yby[8:9:n, :] = np.nan
    
    xbx = np.zeros((npt * 9, n))
    xbx[0:9:n, :] = xlx[:, np.newaxis]
    xbx[1:9:n, :] = xlx[:, np.newaxis]
    xbx[2:9:n, :] = np.nan
    xbx[3:9:n, :] = xlx[:, np.newaxis]
    xbx[4:9:n, :] = xrx[:, np.newaxis]
    xbx[5:9:n, :] = np.nan
    xbx[6:9:n, :] = xrx[:, np.newaxis]
    xbx[7:9:n, :] = xrx[:, np.newaxis]
    xbx[8:9:n, :] = np.nan
    
    ybx = np.zeros((npt * 9, n))
    ybx[0:9:n, :] = ytopx[:, np.newaxis]
    ybx[1:9:n, :] = ybotx[:, np.newaxis]
    ybx[2:9:n, :] = np.nan
    ybx[3:9:n, :] = y[:, np.newaxis]
    ybx[4:9:n, :] = y[:, np.newaxis]
    ybx[5:9:n, :] = np.nan
    ybx[6:9:n, :] = ytopx[:, np.newaxis]
    ybx[7:9:n, :] = ybotx[:, np.newaxis]
    ybx[8:9:n, :] = np.nan
    
    # Plot graph and bars
    hold_state = plt.gca().get_legend_handles_labels()  # Check if hold is on
    h = plt.plot(xby, yby, *varargin[:nVarargs])
    if not hold_state:
        plt.hold(True)
    h += plt.plot(xbx, ybx, *varargin[:nVarargs])
    if not hold_state:
        plt.hold(False)
    if len(h) > 0:
        return h
    
    
    
# Example usage
x = np.arange(1, 11)
y = np.sin(x)
e = np.std(y) * np.ones_like(x)

lux = np.array([e, e])
luy = np.array([e, e])

plt.figure(figsize=(10, 6))
hverrorbar50(x, y, lux, luy, 'o')
plt.title('Error Bar Plot Example')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.show()
