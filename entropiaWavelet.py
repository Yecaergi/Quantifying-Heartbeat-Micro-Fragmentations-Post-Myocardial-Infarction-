import numpy as np
import pywt  # Make sure you have PyWavelets installed for the wavelet functions
from JSDiv import JSDiv
from hverrorbar import hverrorbar
from CotasComplejidad import CotasComplejidad
import scipy.io
import matplotlib.pyplot as plt
import time

def entropiaWavelet(fid, **kwargs): 
    # Default parameters
    mostrar = kwargs.get('mostrar', 0)  # Default value can be adjusted as needed
    wName = kwargs.get('wName', 'gaus6')
    wMaxScale = kwargs.get('wMaxScale', 16)
    wTscale = kwargs.get('wTscale', 'continua')
    procesar = kwargs.get('procesar', 'tiempo')
    subfolder = kwargs.get('subfolder', '')
    recortar = kwargs.get('recortar', False)
     
    # Parse optional arguments
    if 'mostrar' in kwargs:
        mostrar = kwargs['mostrar']
    if 'wName' in kwargs:
        wName = kwargs['wName']
    if 'wMaxScale' in kwargs:
        wMaxScale = kwargs['wMaxScale']
    if 'procesar' in kwargs:
        procesar = kwargs['procesar']
    if 'recortar' in kwargs:
        recortar = kwargs['recortar']
    if 'wTscale' in kwargs:
        wTscale = kwargs['wTscale']
    if 'subfolder' in kwargs:
        subfolder = kwargs['subfolder']
    
    if wTscale == 'discreta':
        wScales = 2 ** np.arange(wMaxScale)       # ok con matlab
    else:
        wScales = np.arange(1, wMaxScale + 1)     # ok con matlab
    
    wNscales = len(wScales)
    
    H = {
    'pd': np.nan,
    'mWd': np.nan,
    'mDd': np.nan,
    'mHd': np.nan,
    'sHd': np.nan,
    'mCd': np.nan,
    'sCd': np.nan,
    'mEt': np.nan,
    'rmsnoise': np.nan,
    'diagnose': np.nan,
    'filas': np.nan,
    'units': np.nan,
    'fileName': np.nan,
    'wName': np.nan,
    'wScales': np.nan,
    'wNscales': np.nan,
    'wMaxScale': np.nan,
    'wTscale': np.nan,
    'procesar': np.nan
}

    
    ar = fid['ondasQRS']
   
    
    units = fid['units'][0]
    sampling = fid['sampling'].item()
   
    H['procesar']=procesar
    H['fileName']=fid['fileName'].item();      # revisado con matlab
   

    
    diagnose = fid['notes'][0][1][4][0][0]
    H['diagnose'] = diagnose                 # revisado con matlab
    
    # Remove NaN-latidos (assuming fid['ondasQRS'] structure)
    hayNaN = np.nanmean(ar, axis=0)
    ar[:,np.isnan(hayNaN)] = [];              # revisado con matlab
    
    #----------------------------------------------------------------------
    #  quito la linea de base a cada onda, la bajo al nivel de la linea de base
    #   de cada onda T
    # ----------------------------------------------------------------------
    columnas, filas = ar.shape                # revisado con matlab
    # Remove baseline from each QRS wave
    ar = ar - np.ones((columnas, 1)) *np.mean(ar, axis=0)  # elimino la media de cada onda

    # Analyze noise     REVISAR Y AGREGARRRRRRRR
   
    VENTANA_RUIDO = int(np.round(20 * sampling / 1000))
    rmsnoise = np.sqrt(np.mean(np.std(ar[0:VENTANA_RUIDO, :], axis=0, ddof=1)**2))
    H['rmsnoise'] = rmsnoise 
    
    
    #--------------------------------------------------------------
    #  proceso tiempo o ensamble
    #--------------------------------------------------------------
    labelx = 'QRS-complexes';
    labely = 'msec';
    if procesar == 'ensamble':
        ar = ar.T
        labelx = 'msec';
        labely = 'QRS-complexes';
        
    columnas, filas = ar.shape
    
    #--------------------------------------------------------------
    # obtengo la escala wavelet de las derivaciones ortogonales
    #--------------------------------------------------------------
  
    swd = np.zeros((wNscales, columnas, filas))
    
    #Truco para incorporar las Db6 en pywt
    ex = pywt.DiscreteContinuousWavelet('db6')
    class DiscreteContinuousWaveletEx(type(ex)):
       def __init__(self, name=u'', filter_bank=None):
           super(type(ex), self)
           pywt.DiscreteContinuousWavelet.__init__(self, name, filter_bank)
           self.complex_cwt = False

    wName = DiscreteContinuousWaveletEx('db6')

    if wTscale == 'discreta':
         sqrt22 = np.sqrt(2) / 2
         for j in range(filas):
             _, sd = pywt.swt(ar[:, j], wName, level=wNscales)
             gain = sqrt22 / np.sqrt(wScales[:, None])
             swd[:, :, j] = gain * np.array(sd)
    else:
         gain = 1. / np.sqrt(wScales[:, np.newaxis])
         # Calcular la CWT para cada fila
         for j in range(filas):
             coef, _ = pywt.cwt(ar[:, j], wScales, wName)
             swd[:, :, j] = gain * coef       #revise y me daba igual   HASTA ACAAAA     
          
    
    
    H['wNscales'] = wNscales;
    H['wMaxScale'] = wMaxScale;
    H['wScales'] = wScales;
    H['wName'] = wName;
    H['wTscale'] = wTscale;        
   
   #--------------------------------------------------------------------------------------------------
   # me quedo unicamente con el QRS para procesar
   #---------------------------------------------------------------------------
    
    if recortar:
        ini = int(np.round((fid['resQRS']['Qp'][0][0][0][0] - 10) * sampling / 1000))
        fin = int(np.round((fid['resQRS']['Sp'][0][0][0][0] + 10) * sampling / 1000))
  
        if ini < 1:
            ini = 1
        if procesar == 'tiempo':
            if fin > columnas:
                fin = columnas
            ar = ar[ini-1:fin, :]
            swd = swd[:, ini-1:fin, :]
        else:
            if fin > filas:
                fin = filas
            ar = ar[:, ini-1:fin]
            swd = swd[:, :, ini-1:fin]
            
       
                
    columnas, filas=ar.shape
   

    #----------------------------------------------------------------------
    #  Analizo la entropia completa de cada onda
    #----------------------------------------------------------------------
    
    # (I) energia escalas x latidos
    Ed = np.squeeze(np.sum(np.abs(swd) ** 2, axis=1))

    # (II) energia total x filas
    Et = np.sum(Ed, axis=0)
    
    # (III) distribucion de probabilidad escalas x latidos
    pd = Ed / (np.ones((wNscales,1)) * Et)
  
    # (IV) entropia total x latidos
    Wd = - np.sum(pd * np.log(pd + np.finfo(float).eps), axis=0)
    # (V) desorden x latidos (entropia normalizada). [0 1]
    Hd = Wd / np.log(wNscales + np.finfo(float).eps)

    
    Q0 = -2 / (((wNscales + 1) / wNscales) * np.log(wNscales + 1) - 2 * np.log(2 * wNscales) + np.log(wNscales))
    # (VI) desequilibrio euclideo x latidos
    # Dd = wNscales/(wNscales-1) * sum((pd - 1/wNscales).^2, 1);
    # (VI) desequilibrio de Jansen-Shannon x latidos
    Dd = Q0 * JSDiv(pd, 1 / wNscales)  
    
    Cd = Hd * Dd
   


    # segun Rosso, 2006
    # (I)  ec. 37. Energía en cada escala es la sum_t |S(j,t)|^2, con j escalas
    #      a lo largo del tiempo t. Obtengo J valores de energía, uno x escala.
    # (II) ec. 38. Energía total, es la suma de las energías de las escalas.
    # (III)ec. 39. Energia relativa x escala, o densidad de probabilidad pd.
    # (IV) ec. 40. Entropia global por latido, a lo largo de las escalas, es un
    #      único valor x onda, esto es, por conjunto de J escalas de una onda.
    # (V)  ecs. 1,9. Desorden es entropia normalizada con uniforme 1/J escalas:
    #      -sum_J (1/J *log(1/J)) = log(J), luego H = W / log(J).
    # (VI) ecs. 2,11. Desequilibrio es la separacion pd respecto a una uniforme,
    #      como pd es a lo largo de las escalas, la normal es 1/J escalas.
    #      Esta normalizada entre [0 1] segun Lamberti (2004)
    # (VII)ecs. 3,41. Complejidad. Valor unico x onda, o sea por J escalas.
    H['pd'] = np.mean(pd, axis=1, keepdims=True)
    H['mEt'] = np.mean(Et)     # energia media de los latidos
    H['mWd'] = np.mean(Wd)     # entropia media de los latidos
    H['mHd'] = np.mean(Hd)     # desorden medio (entropia normalizada) de los latidos
    H['sHd'] = np.std(Hd)      # desorden SD (entropia normalizada) de los latidos
    H['mDd'] = np.mean(Dd)     # desequilibrio medio de los latidos
    H['mCd'] = np.mean(Cd)     # complejidad media de los latidos
    H['sCd'] = np.std(Cd)      # complejidad SD de los latidos

    wdFilas = len(Wd);    
    H['filas'] = wdFilas
    
    #----------------------------------------------------------------------
    # grafico
    #----------------------------------------------------------------------
    fontsize = 12

    def graficar():
     if mostrar == 3:
      
       fig = plt.figure(1)
       fig.suptitle('Waves: ' + H['fileName'], fontsize=12)
       # Generar datos para el mesh
       arX = np.arange(1, filas + 1)
       arY = np.arange(1, columnas + 1) * 1000 / sampling
       ax = fig.add_subplot(111, projection='3d')
       X, Y = np.meshgrid(arX, arY)
       mesh = ax.plot_surface(X, Y, ar, cmap='viridis')
       ax.view_init(0, 90)
       ax.view_init(-75, 30)
       ax.tick_params(axis='both', which='major', labelsize=10)
       ax.invert_yaxis()
       ax.set_xlabel(labelx, fontsize=10)
       ax.set_ylabel(labely, fontsize=10)
       ax.set_zlabel(f'Amplitude [{units}]', fontsize=10)
       ax.set_xlim([1, filas])
       ax.set_ylim([0, (columnas - 1) * 1000 / sampling])
       plt.show()
       pass
   
     elif mostrar == 4:
        fig, ax = plt.subplots()
        fig.suptitle('Waves: ' + H['fileName'], fontsize=12)
        ax.plot(H['Hd'], H['Cd'], color='b', linestyle='dotted')
        ax.plot(H['mHd'], H['mCd'], color='k', marker='o', markersize=6, markerfacecolor='k')
        CotasComplejidad(wNscales, 'cualentropia', 'SHANNON', 'isplot', True)
        xx=[np.mean(Hd)]
        yy=[np.mean(Cd)]       
        lux = np.std(Hd) * np.array([[1], [1]])
        luy=np.std(Cd) * np.array([[1], [1]])
        hverrorbar(xx, yy, lux, luy, color='k', linestyle='solid', linewidth=1)  
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.legend(['QRS-complexes'])
        plt.xlim([0.7, 1])
        plt.ylim([0, 0.25])
        plt.show()
        pass
    
     elif mostrar == 5:
        plt.figure('Waves: ' + 'fileName', figsize=(10, 6))
        mean_pd = np.mean(pd, axis=1)
        std_pd = np.std(pd, axis=1)
        plt.semilogy(wScales, mean_pd, linestyle='none', color='k',
             markerfacecolor='k', marker='o', markersize=5)
        plt.errorbar(wScales, mean_pd, yerr=std_pd, fmt='k', linestyle='none', linewidth=1)
        plt.xlabel('Resolution levels', fontsize=fontsize)
        plt.ylabel('Relative wavelet energy', fontsize=fontsize)
        plt.xlim([0, wScales[-1] + 1])
        plt.xticks([1, 6, 11, 16])
        plt.ylim([0.003, 0.4])
        plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)
        plt.show()
        pass
    
     elif mostrar == 6:
        fig = plt.figure(num='Entropia - ' + 'fileName', figsize=(10, 6))
        backend = plt.get_backend()

        fig.set_size_inches(2.4 * fig.get_size_inches()[0], 1.2 * fig.get_size_inches()[1])


        plt.subplot(121)
        pHd = np.polyfit(Hd, Dd, 1)
        xi = np.linspace(min(Hd), max(Hd))
        yi = pHd[0] * xi + pHd[1]
        plt.loglog(Hd, Dd, linestyle='none', color='r', marker='o')
        plt.plot(xi, yi, color='b', linewidth=1)
        plt.xlabel('Entropia normalizada [n.u.]', fontsize=fontsize)
        plt.ylabel('Desequilibrio [n.u.]', fontsize=fontsize)
        plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)

        plt.subplot(122)
        pHd = np.polyfit(Hd, Et, 1)
        yi = pHd[0] * xi + pHd[1]
        plt.loglog(Hd, Et, linestyle='none', color='r', marker='o')
        plt.plot(xi, yi, color='b', linewidth=1)
        plt.xlabel('Entropia normalizada [n.u.]', fontsize=fontsize)
        plt.ylabel(f'Energia [{units}^2]', fontsize=fontsize)
        plt.gca().tick_params(axis='both', which='major', labelsize=fontsize)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.suptitle(f'{diagnose} - {wName} {wTscale} {wNscales} escalas')
        plt.show()
        pass
    
    return H



#mat_contents=scipy.io.loadmat('patient268_v5.mat');

#xx = mat_contents['ondasQRS'];

#resultado = entropiaWavelet(mat_contents, mostrar=6, wName='gaus6', wMaxScale= 16, wTscale='continua', procesar='tiempo', subfolder=' ', recortar=True);



