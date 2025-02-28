import numpy as np
from ptbaprocesar import PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF, PTBMI07VTVF, PTBMI60VTVF, PTBMI07NOVTVFANT, PTBMI60NOVTVFANT, PTBMI07NOVTVFINF, PTBMI60NOVTVFINF, ALLPTB
from dL import dL
import scipy.io
import matplotlib.pyplot as plt
from hverrorbar import hverrorbar
from CotasComplejidad import CotasComplejidad
from myroc11 import MyROC
from getPsymbol import getPsymbol
from scipy.stats import ttest_ind
from miAnova import miAnova
from getNote import getNote




# ------------------------------------------------------------------------
# si muestro resultados o gráfico algo...
# ------------------------------------------------------------------------
graficar = 1

# ------------------------------------------------------------------------
# Abro el archivo donde se encuentran los TWSV de las ondas T
# ------------------------------------------------------------------------
cualResNombre = 0
ResNombre = [
    '\\ctrl-healing-healed sin VTVF QRS Entropia PTB.mat',
    '\\ctrl-healing-healed sin VTVF lineaBase Entropia.mat',
    '\\ctrl-healing-healed sin VTVF ondaT Entropia.mat'
]
Nombre = ResNombre[cualResNombre]

cualsubfolder = ['QRS', 'lineaBase', 'ondaT']
subfolder = cualsubfolder[cualResNombre]

Carpeta = 'Resultados'

# Cargar archivos .mat
data = scipy.io.loadmat(f'{Carpeta}/{Nombre}')
#bl = scipy.io.loadmat(f'{Carpeta}/{ResNombre[1]}')
ptbaprocesar = scipy.io.loadmat('ptbaprocesar.mat')

# Parámetros a comparar
paramx = 'mHd'
paramy = 'mCd'



# -----------------------------------------------------------------------
# Solo me quedo con todos los infartos, anteriores o inferiores
# -----------------------------------------------------------------------
cualfiltrar = 0
filtrarM7 = [ptbaprocesar['PTBMI07NOVTVF'], ptbaprocesar['PTBMI07NOVTVFANT'], ptbaprocesar['PTBMI07NOVTVFINF']]
filtrarM60 = [ptbaprocesar['PTBMI60NOVTVF'], ptbaprocesar['PTBMI60NOVTVFANT'], ptbaprocesar['PTBMI60NOVTVFINF']]



# Filtrar grupo2
for idx in range(len(data['grupo2Name']) - 1, -1, -1):
    if not np.any(np.isin(filtrarM7[cualfiltrar], data['grupo2Name'][idx])):
        data['grupo2Name'] = np.delete(data['grupo2Name'], idx)
        data['grupo2OriginalName'] = np.delete(data['grupo2OriginalName'], idx)
        data['Hgrupo2'] = np.delete(data['Hgrupo2'], idx, axis=0)
        data['nGrupo2'] = data['Hgrupo2'].shape[0]
        data['notes2'] = np.delete(data['notes2'], idx)
        data['rmsnoisegrupo2'] = np.delete(data['rmsnoisegrupo2'], idx, axis=0)
        
# Filtrar grupo3
for idx in range(len(data['grupo3Name']) - 1, -1, -1):
    if not np.any(np.isin(filtrarM60[cualfiltrar], data['grupo3Name'][idx])):
        data['grupo3Name'] = np.delete(data['grupo3Name'], idx)
        data['grupo3OriginalName'] = np.delete(data['grupo3OriginalName'], idx)
        data['Hgrupo3'] = np.delete(data['Hgrupo3'], idx, axis=0)
        data['nGrupo3'] = data['Hgrupo3'].shape[0]
        data['notes3'] = np.delete(data['notes3'], idx)
        data['rmsnoisegrupo3'] = np.delete(data['rmsnoisegrupo3'], idx, axis=0)



Hgrupo1 = data['Hgrupo1']
Hgrupo2 = data['Hgrupo2']
Hgrupo3 = data['Hgrupo3']

G1_NAME='CTRL'
G2_NAME='MI7'
G3_NAME='MI60'

grupo1Name = data['grupo1Name']
grupo2Name = data['grupo2Name']
grupo3Name = data['grupo3Name']

grupo1OriginalName = data['grupo1OriginalName']
grupo2OriginalName = data['grupo2OriginalName']
grupo3OriginalName = data['grupo3OriginalName']

notes1 = data['notes1']
notes2 = data['notes2']
notes3 = data['notes3']

print(notes1)

rmsnoisegrupo1 = data['rmsnoisegrupo1']
rmsnoisegrupo2 = data['rmsnoisegrupo2']
rmsnoisegrupo3 = data['rmsnoisegrupo3']

nGrupo1 = Hgrupo1.shape[0]
nGrupo2 = Hgrupo2.shape[0]
nGrupo3 = Hgrupo3.shape[0]



strcualfiltrar = ['Todos los infartados', 'Infarto anteriores', 'Infarto inferiores']

multilead = [
    list(range(dL.I, dL.V6 + 1)),
    [dL.I, dL.AVL] + list(range(dL.V1, dL.V6 + 1)),
    [dL.II, dL.III, dL.AVF, dL.V5, dL.V6]
]

leadNames = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6','ML']
leads = list(range(13))
NLEADS = len(leads)
ML = multilead[cualfiltrar]

wName = Hgrupo1[0][0]['wName'][0][0][0]  # O [0] si es una matriz en lugar de un array
wScales = Hgrupo1[0][0]['wScales'][0][0][0]
wNscales = Hgrupo1[0][0]['wNscales'][0][0][0][0]
wTscale = Hgrupo1[0][0]['wTscale'][0][0][0]
units = wTscale = Hgrupo1[0][0]['units'][0][0][0][0]

# Translate conditions for paramx
if paramx == 'mEt':
    xlimite = [0, 2]
    xstrparam = f'Energy [{units}^{2}]'
elif paramx == 'mWd':
    xlimite = [1.5, 3]
    xstrparam = 'Whole wavelet entropy'
elif paramx == 'mHd':
    xlimite = [0.6, 1.1]
    xstrparam = 'Normalized wavelet entropy'
elif paramx == 'mDd':
    xlimite = [0, 0.2]
    xstrparam = 'Desequilibrium [n.u.]'
elif paramx == 'mCd':
    xlimite = [0, 0.3]
    xstrparam = 'Wavelet statistical complexity'
elif paramx == 'filas':
    xlimite = [-float('inf'), float('inf')]
    xstrparam = 'beats'

# Translate conditions for paramy
if paramy == 'mEt':
    ylimite = [0, 2]
    ystrparam = f'Energy [{units}^{2}]'
elif paramy == 'mWd':
    ylimite = [1.5, 3]
    ystrparam = 'Whole wavelet entropy'
elif paramy == 'mHd':
    ylimite = [0.6, 1.1]
    ystrparam = 'Normalized wavelet entropy'
elif paramy == 'mDd':
    ylimite = [0, 0.2]
    ystrparam = 'Desequilibrium [n.u.]'
elif paramy == 'mCd':
    ylimite = [0, 0.3]
    ystrparam = 'Wavelet statistical complexity'
elif paramy == 'filas':
    ylimite = [-float('inf'), float('inf')]
    ystrparam = 'beats'

# Declaración de variables
ng1 = np.full(NLEADS, np.nan)
ng2 = np.full(NLEADS, np.nan)
ng3 = np.full(NLEADS, np.nan)

nc1 = np.full(NLEADS, np.nan)
nc2 = np.full(NLEADS, np.nan)
nc3 = np.full(NLEADS, np.nan)

mg1 = np.full(NLEADS, np.nan)
mg2 = np.full(NLEADS, np.nan)
mg3 = np.full(NLEADS, np.nan)

sem1 = np.full(NLEADS, np.nan)
sem2 = np.full(NLEADS, np.nan)
sem3 = np.full(NLEADS, np.nan)

Semg1 = np.full((2, NLEADS), np.nan)
Semg2 = np.full((2, NLEADS), np.nan)
Semg3 = np.full((2, NLEADS), np.nan)

Semc1 = np.full((2, NLEADS), np.nan)
Semc2 = np.full((2, NLEADS), np.nan)
Semc3 = np.full((2, NLEADS), np.nan)

mc1 = np.full(NLEADS, np.nan)
mc2 = np.full(NLEADS, np.nan)
mc3 = np.full(NLEADS, np.nan)

Pg12 = np.full(NLEADS, np.nan)
Pc12 = np.full(NLEADS, np.nan)
sigg12 = [None] * NLEADS
sigc12 = [None] * NLEADS
P2D12 = np.full(NLEADS, np.nan)
sig2D12 = [None] * NLEADS

Pg13 = np.full(NLEADS, np.nan)
Pc13 = np.full(NLEADS, np.nan)
sigg13 = [None] * NLEADS
sigc13 = [None] * NLEADS
P2D13 = np.full(NLEADS, np.nan)
sig2D13 = [None] * NLEADS

Pg23 = np.full(NLEADS, np.nan)
Pc23 = np.full(NLEADS, np.nan)
sigg23 = [None] * NLEADS
sigc23 = [None] * NLEADS
P2D23 = np.full(NLEADS, np.nan)
sig2D23 = [None] * NLEADS

stgMI7 = {}
stcMI7 = {}
stgMI60 = {}
stcMI60 = {}
senMI7 = {}
espMI7 = {}
senMI60 = {}
espMI60 = {}
stjointMI7 = {}
stjointMI60 = {}
senjMI7 = {}
espjMI7 ={}
senjMI60 = {}
espjMI60 = {}
  

#------------------------------------------------------------------
RMSNOISE = 0.02  # en [mV]
scale = 2

# Boxplot datos se acumulan acá
grubox, mgrubox, posbox = [], [], []
quanbox, sigbox = [], []
k = 1

n1 = len(grupo1Name)
n2 = len(grupo2Name)
n3 = len(grupo3Name)

m = len(leads) 
g1w = np.zeros((n1, m))
g2w = np.zeros((n2, m))
g3w = np.zeros((n3, m))
c1w = np.zeros((n1, m))
c2w = np.zeros((n2, m))
c3w = np.zeros((n3, m))

for j in range(n1):
    for i in range(m):
        g1w[j, i] = Hgrupo1[j][leads[i]]['mHd'][0][0][0][0]
        c1w[j, i] = Hgrupo1[j][leads[i]]['mCd'][0][0][0][0]

for j in range(n2):
    for i in range(m):
        g2w[j, i] = Hgrupo2[j][leads[i]]['mHd'][0][0][0][0]
        c2w[j, i] = Hgrupo2[j][leads[i]]['mCd'][0][0][0][0]
        
for j in range(n3):
    for i in range(m):
        g3w[j, i] = Hgrupo3[j][leads[i]]['mHd'][0][0][0][0]
        c3w[j, i] = Hgrupo3[j][leads[i]]['mCd'][0][0][0][0]
        
g1w = g1w[:, 1:13]   #saque en todos los casos la columna 0 q era de nan
g2w = g2w[:, 1:13]
g3w = g3w[:, 1:13]
c1w = c1w[:, 1:13]
c2w = c2w[:, 1:13]
c3w = c3w[:, 1:13]    

ML_adjusted = [index - 1 for index in ML]

rmsnoisegrupo1 = rmsnoisegrupo1[:, 1:13]
rmsnoisegrupo2 = rmsnoisegrupo2[:, 1:13]
rmsnoisegrupo3 = rmsnoisegrupo3[:, 1:13]



for i in range(len(leads)):
    if i < NLEADS-1:
        g1 = g1w[:, i]
        g2 = g2w[:, i]
        g3 = g3w[:, i]
        c1 = c1w[:, i]
        c2 = c2w[:, i]
        c3 = c3w[:, i]
        noise1 = np.nanmean(rmsnoisegrupo1[:, i])
        noise2 = np.nanmean(rmsnoisegrupo2[:, i])
        noise3 = np.nanmean(rmsnoisegrupo3[:, i])
    else:
        g1 = np.sqrt(np.nanmean(g1w[:, ML_adjusted]**2, axis=1))
        g2 = np.sqrt(np.nanmean(g2w[:, ML_adjusted]**2, axis=1))
        g3 = np.sqrt(np.nanmean(g3w[:, ML_adjusted]**2, axis=1))
        c1 = np.sqrt(np.nanmean(c1w[:, ML_adjusted]**2, axis=1))
        c2 = np.sqrt(np.nanmean(c2w[:, ML_adjusted]**2, axis=1))
        c3 = np.sqrt(np.nanmean(c3w[:, ML_adjusted]**2, axis=1))
        noise1 = np.nanmean(rmsnoisegrupo1[:, ML_adjusted], axis=1)
        noise2 = np.nanmean(rmsnoisegrupo2[:, ML_adjusted], axis=1)
        noise3 = np.nanmean(rmsnoisegrupo3[:, ML_adjusted], axis=1)
        
    noise1 = np.where(noise1 > RMSNOISE, np.nan, noise1)
    noise2 = np.where(noise2 > RMSNOISE, np.nan, noise2)
    noise3 = np.where(noise3 > RMSNOISE, np.nan, noise3)

    # Remove NaN values from arrays
    g1 = g1[~np.isnan(noise1)]
    g2 = g2[~np.isnan(noise2)]
    g3 = g3[~np.isnan(noise3)]
    c1 = c1[~np.isnan(noise1)]
    c2 = c2[~np.isnan(noise2)]
    c3 = c3[~np.isnan(noise3)]
    
    # Further remove NaN values directly from c1, c2, and c3
  
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]
    g3 = g3[~np.isnan(g3)]
    c1 = c1[~np.isnan(c1)]
    c2 = c2[~np.isnan(c2)]
    c3 = c3[~np.isnan(c3)]
    
    g1 = np.round(g1,4)
    g2 = np.round(g2,4)
    g3 = np.round(g3,4)
    c1 = np.round(c1,4)
    c2 = np.round(c2,4)
    c3 = np.round(c3,4)
    
    noise1 = noise1[~np.isnan(noise1)]
    noise2 = noise2[~np.isnan(noise2)]
    noise3 = noise3[~np.isnan(noise3)]

    ng1[i] = len(g1)
    ng2[i] = len(g2)
    ng3[i] = len(g3)
    nc1[i] = len(c1)
    nc2[i] = len(c2)
    nc3[i] = len(c3)    
    
    # t-test para muestras independientes con medias iguales
    _, Pg12[i] = ttest_ind(g1, g2)
    _, Pg13[i] = ttest_ind(g1, g3)
    _, Pg23[i] = ttest_ind(g2, g3)
    sigg12[i], sigg13[i], sigg23[i] = getPsymbol(Pg12[i], Pg13[i], Pg23[i])

    _, Pc12[i] = ttest_ind(c1, c2)
    _, Pc13[i] = ttest_ind(c1, c3)
    _, Pc23[i] = ttest_ind(c2, c3)
    sigc12[i], sigc13[i], sigc23[i] = getPsymbol(Pc12[i], Pc13[i], Pc23[i])

    mg1[i] = np.nanmean(g1)  # CALCULO LA MEDIA Y SEM
    mg2[i] = np.nanmean(g2)
    mg3[i] = np.nanmean(g3)
    mc1[i] = np.nanmean(c1)
    mc2[i] = np.nanmean(c2)  
    mc3[i] = np.nanmean(c3)    
    
    Semg1 = np.nanstd(g1)*np.array([[1], [1]])
    Semg2 = np.nanstd(g2)*np.array([[1], [1]])
    Semg3 = np.nanstd(g3)*np.array([[1], [1]])
    Semc1 = np.nanstd(c1)*np.array([[1], [1]])
    Semc2 = np.nanstd(c2)* np.array([[1], [1]])
    Semc3 = np.nanstd(c3)* np.array([[1], [1]])
    
    if i < NLEADS:
    # Concateno los datos para el boxplot. Para tener separado por ternas,
    # agrego una columna adicional NaN entre ternas, de esta manera, al
    # mostrarse el boxplot, aparece un espacio vacío. Pero desde el
    # algoritmo hay que considerarlo como un dato más y tenerlo en cuenta.
    
       grubox = np.concatenate((grubox, g1, g2, g3, [np.nan]))  # datos del eje y, los boxes
       mgrubox = np.concatenate((mgrubox, [mg1[i], mg2[i], mg3[i], np.nan]))  # media de cada box
       quanbox = np.concatenate((quanbox, [np.quantile(g1, 1), np.quantile(g2, 1), np.quantile(g3, 1), np.nan]))
       posbox = np.concatenate((posbox, np.full(int(ng1[i]), k), 
                                  np.full(int(ng2[i]), k+1), 
                                  np.full(int(ng3[i]), k+2), 
                                  [k+3]))

    
       # Símbolos significancia, mi7 y m60 los agrupo en un solo string
       sigbox += f" {sigg12[i]} {sigg13[i]}{sigg23[i]} "
       k += 4  # incremento de a 4: 3 por los grupos + 1 por el espacio vacío

    stgMI7[i] = MyROC.ROC(g1, g2)  
    stcMI7[i] = MyROC.ROC(c1, c2)
    stgMI60[i] = MyROC.ROC(g1, g3)  
    stcMI60[i] = MyROC.ROC(c1, c3)
    
    # Extrae los valores de sensibilidad (SEN) y especificidad (ESP) de los resultados ROC
    senMI7[i] = stgMI7[i]['SEN']  # TP/(TP+FN)
    espMI7[i] = stgMI7[i]['ESP']
    senMI60[i] = stgMI60[i]['SEN']
    espMI60[i] = stgMI60[i]['ESP']
    
   # Hacerlo asi porque sino no da!!!
    stjointMI7[i] = MyROC.jROC(np.column_stack([g1, c1]), np.column_stack([g2, c2]))
    stjointMI60[i] = MyROC.jROC(np.column_stack([g1, c1]), np.column_stack([g3, c3]))

    # Extrae los valores de sensibilidad (SEN) y especificidad (ESP) de los resultados jROC
    senjMI7[i] = stjointMI7[i]['SEN']  # TP/(TP+FN)
    espjMI7[i] = stjointMI7[i]['ESP']
    senjMI60[i] = stjointMI60[i]['SEN']
    espjMI60[i] = stjointMI60[i]['ESP']
    
    
 
    
    if graficar == 1:
       print(f"Analizando {leadNames[i]} ...")

       print(f"{xstrparam}\nCTRL: {mg1[i]:.4f}, MI7: {mg2[i]:.4f}({sigg12[i]},{Pg12[i]:.4f}), "
          f"MI60: {mg3[i]:.4f}({sigg13[i]},{Pg13[i]:.4f}; {sigg23[i]},{Pg23[i]:.4f})")

       print(f"CTRL-MI7  G: sens: {stgMI7[i]['SEN']:.0f}%, espec: {stgMI7[i]['ESP']:.0f}%, AUC: {stgMI7[i]['AUC']:.2f}")
       print(f"CTRL-MI60 G: sens: {stgMI60[i]['SEN']:.0f}%, espec: {stgMI60[i]['ESP']:.0f}%, AUC: {stgMI60[i]['AUC']:.2f}\n")

       print(f"{ystrparam}\nCTRL: {mc1[i]:.4f}, MI7: {mc2[i]:.4f}({sigc12[i]},{Pc12[i]:.4f}), "
          f"MI60: {mc3[i]:.4f}({sigc13[i]},{Pc13[i]:.4f}; {sigc23[i]},{Pc23[i]:.4f})")

       print(f"CTRL-MI7  C: sens: {stcMI7[i]['SEN']:.0f}%, espec: {stcMI7[i]['ESP']:.0f}%, AUC: {stcMI7[i]['AUC']:.2f}")
       print(f"CTRL-MI60 C: sens: {stcMI60[i]['SEN']:.0f}%, espec: {stcMI60[i]['ESP']:.0f}%, AUC: {stcMI60[i]['AUC']:.2f}\n")

       print("Sensibilidad y especificidad conjunta")
       print(f"CTRL-MI7  CjG: sens: {stjointMI7[i]['SEN']:.0f}%, espec: {stjointMI7[i]['ESP']:.0f}%, AUC: {stjointMI7[i]['AUC']:.2f}")
       print(f"CTRL-MI60 CjG: sens: {stjointMI60[i]['SEN']:.0f}%, espec: {stjointMI60[i]['ESP']:.0f}%, AUC: {stjointMI60[i]['AUC']:.2f}\n")
       
#------------------------------------------------------------------
wNscales = 16

if graficar == 1:
    # Grafico los resultados de entropía-complejidad
    if i == leads[-1]:
        # Análisis de varianza entre-sujetos para muestras repetidas.
        # Uso de muestras repetidas porque tanto W como C se obtienen a partir
        # de los mismos datos, es decir, los participantes (pacientes)
        # participan tanto en el cálculo de W como en el de C.
        # Utilizar dos con una sola variable a analizar es lo mismo que 
        # ttest2, para dos o más grupos o variables a analizar, usar ANOVA
        resTbl12 = miAnova(np.column_stack([g1, c1]), np.column_stack([g2, c2]), [G1_NAME, G2_NAME], [paramx, paramy])
        resTbl13 = miAnova(np.column_stack([g1, c1]), np.column_stack([g3, c3]), [G1_NAME, G3_NAME], [paramx, paramy])
        resTbl23 = miAnova(np.column_stack([g2, c2]), np.column_stack([g3, c3]), [G2_NAME, G3_NAME], [paramx, paramy])

        # Supongamos que `resTbl12` contiene los resultados del análisis de 'mHd' y 'mCd'
        P2D12[i] = resTbl12['mHd']['PR(>F)'].iloc[0]
        P2D13[i] = resTbl12['mCd']['PR(>F)'].iloc[0]
        P2D23[i] = resTbl13['mHd']['PR(>F)'].iloc[0]  # Ajustar según la estructura

        
        sig2D12[i], sig2D13[i], sig2D23[i] = getPsymbol(P2D12[i], P2D13[i], P2D23[i])
        fig, ax = plt.subplots(1)
        fsize = 14
        wcolor = 'white'
        lgcolor = 'grey'
        gcolor = 'lightgrey'
        hverrorbar(mg1[i], mc1[i], Semg1, Semc1, color='k', linestyle='-', linewidth=1)
        hverrorbar(mg2[i], mc2[i], Semg2, Semc2, color='k', linestyle='-', linewidth=1)
        hverrorbar(mg3[i], mc3[i], Semg3, Semc3, color='k', linestyle='-', linewidth=1)
        hL1 =ax.plot(mg1[i], mc1[i], 'ks', markerfacecolor=wcolor, markersize=8, label='CTRL')
        hL2 =ax.plot(mg2[i], mc2[i], 'ko', markerfacecolor=lgcolor, markersize=8, label='MI7')
        hL3 =ax.plot(mg3[i], mc3[i], 'kd', markerfacecolor=gcolor, markersize=8, label='MI60')
        # Ajustes del gráfico
        ax.set_xlabel(r'normalized entropy Hml', fontsize=fsize+3)
        ax.set_ylabel(r'wavelet complexity Cml', fontsize=fsize+3)
        # Placeholder for CotasComplejidad function. Implement the function as needed.
        CotasComplejidad(wNscales, 'k-.', 'cualentropia', 'SHANNON')
        # Inicializa la cadena `signif` concatenando `sig2D12[i]` y `sig2D13[i]`
        signif = f"  {sig2D12[i]}, {sig2D13[i]}"

       # Ajusta el tamaño de fuente de la gráfica actual
        plt.gca().tick_params(labelsize=fsize)

        ax.set_xlim(0.7, 0.95)
        ax.set_ylim(0.1, 0.25)
        ax.set_xticks(np.arange(0.7, 0.95, 0.05))
        ax.set_yticks(np.arange(0.1, 0.25, 0.05))
        ax.legend(loc='upper right', fontsize=fsize)
        fig.canvas.manager.set_window_title(strcualfiltrar[cualfiltrar])
        plt.grid(False)
        plt.show()
        
        
        

