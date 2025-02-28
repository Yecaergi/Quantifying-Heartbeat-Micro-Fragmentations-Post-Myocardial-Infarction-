import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, f1_score
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
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, f_oneway, kruskal
from scipy.stats import fisher_exact
from scipy.stats import shapiro
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats
from scipy.spatial.distance import cdist
from itertools import combinations
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix,
    f1_score
)


def cross_validate(features, labels, patient_ids, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    sensitivities, specificities, aucs, f1_scores = [], [], [], []
    fold_patient_info = []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Optimización del umbral
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (y_pred_proba > optimal_threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        aucs.append(auc)
        f1_scores.append(f1)

        # Almacenar información de los pacientes testeados
        fold_patient_info.append({
            'fold': len(fold_patient_info) + 1,
            'patient_ids': patient_ids[test_index],
            'labels': y_test
        })

    # Agregar cálculos de media, mínimo y std para sensibilidad, especificidad y AUC
    return {
        'sens_mean': np.mean(sensitivities),
        'sens_min': np.min(sensitivities),
        'sens_std': np.std(sensitivities),
        'spec_mean': np.mean(specificities),
        'spec_min': np.min(specificities),
        'spec_std': np.std(specificities),
        'auc_mean': np.mean(aucs),
        'auc_min': np.min(aucs),
        'auc_std': np.std(aucs),
        'f1_mean': np.mean(f1_scores),
        'fold_patient_info': fold_patient_info
    }

# Definición de las funciones
def calc_observed_stat(data, labels):
    # Seleccionamos solo las columnas numéricas: 'Ent' y 'Comp'
    numeric_data = data[['Ent', 'Comp']]
    
    # Agrupar por grupo y calcular las medias
    means = numeric_data.groupby(labels).mean().values
    
    # Calcular la distancia euclidiana entre las medias de los grupos
    distance_matrix = cdist(means, means, metric='euclidean')
    return np.sum(distance_matrix)  # Estadística observada

def permute_and_calc_stat(data, labels, n_permutations=1000):
    permuted_stats = []
    for _ in range(n_permutations):
        permuted_labels = np.random.permutation(labels)
        stat = calc_observed_stat(data, permuted_labels)
        permuted_stats.append(stat)
    return np.array(permuted_stats)

def compare_groups_posthoc(data, group_1, group_2, n_permutations=1000):
    # Seleccionar los datos de los dos grupos
    data_group_1 = data[data['Grupo'] == group_1]
    data_group_2 = data[data['Grupo'] == group_2]
    
    # Etiquetas de los grupos para la comparación
    labels_group_1 = np.array([0] * len(data_group_1))
    labels_group_2 = np.array([1] * len(data_group_2))
    
    # Concatenar los datos de los dos grupos
    data_combined = pd.concat([data_group_1, data_group_2], axis=0)
    labels_combined = np.concatenate([labels_group_1, labels_group_2])
    
    # Realizar las permutaciones
    permuted_stats = permute_and_calc_stat(data_combined, labels_combined, n_permutations)
    
    # Calcular la estadística observada para el par de grupos
    observed_stat = calc_observed_stat(data_combined, labels_combined)
    
    # Calcular el valor p
    p_value = np.sum(permuted_stats >= observed_stat) / n_permutations
    
    return observed_stat, p_value

# ------------------------------------------------------------------------
# Flag para graficar resultados
graficar = 1

# ------------------------------------------------------------------------
# Nombres de los archivos PTB y PTBXL
ResNombre = [
    '\\ctrl-healing-healed sin VTVF QRS Entropia PTB.mat',
    '\\ctrl-healing-healed sin VTVF QRS Entropia PTB XL.mat'
]

Carpeta = 'Resultados'

# Cargar archivos .mat de PTB y PTBXL
data_ptb = scipy.io.loadmat(f'{Carpeta}/{ResNombre[0]}')
data_ptbxl = scipy.io.loadmat(f'{Carpeta}/{ResNombre[1]}')

# Cargar archivo con información de pacientes
ptbaprocesar = scipy.io.loadmat('ptbaprocesar11.mat')



# Parámetros a comparar
paramx = 'mHd'
paramy = 'mCd'

# -----------------------------------------------------------------------
# Filtrar grupos de pacientes (infartos anteriores o inferiores)
cualfiltrar = 0
filtrarM7 = [ptbaprocesar['PTBMI07NOVTVF'], ptbaprocesar['PTBMI07NOVTVFANT'], ptbaprocesar['PTBMI07NOVTVFINF']]
filtrarM60 = [ptbaprocesar['PTBMI60NOVTVF'], ptbaprocesar['PTBMI60NOVTVFANT'], ptbaprocesar['PTBMI60NOVTVFINF']]

# Filtrar grupos en el archivo combinado
def filtrar_grupos(grupoName, filtro):
    for idx in range(len(grupoName) - 1, -1, -1):
        if not np.any(np.isin(filtro, grupoName[idx])):
            grupoName = np.delete(grupoName, idx)
    return grupoName

# Procesamiento de datos para el análisis y filtrado (igual que antes)
Hgrupo1 = np.concatenate((data_ptb['Hgrupo1'], data_ptbxl['Hgrupo1']), axis=0)
Hgrupo2 = np.concatenate((data_ptb['Hgrupo2'], data_ptbxl['Hgrupo2']), axis=0)
Hgrupo3 = np.concatenate((data_ptb['Hgrupo3'], data_ptbxl['Hgrupo3']), axis=0)

# Otros parámetros y configuraciones (igual que antes)
G1_NAME='CTRL'
G2_NAME='MI7'
G3_NAME='MI60'

grupo1Name =np.concatenate((data_ptb['grupo1Name'], data_ptbxl['grupo1Name']), axis=0)
grupo2Name = np.concatenate((data_ptb['grupo2Name'], data_ptbxl['grupo2Name']), axis=0)
grupo3Name = np.concatenate((data_ptb['grupo3Name'], data_ptbxl['grupo3Name']), axis=0)
# Concatenar los grupos
todos_los_grupos = np.concatenate((grupo1Name, grupo2Name, grupo3Name), axis=0)

grupo1OriginalName = np.concatenate((data_ptb['grupo1OriginalName'], data_ptbxl['grupo1OriginalName']), axis=0)
grupo2OriginalName = np.concatenate((data_ptb['grupo2OriginalName'], data_ptbxl['grupo2OriginalName']), axis=0)
grupo3OriginalName = np.concatenate((data_ptb['grupo3OriginalName'], data_ptbxl['grupo3OriginalName']), axis=0)

notes1_ptb = data_ptb['notes1']
notes2_ptb = data_ptb['notes2']
notes3_ptb = data_ptb['notes3']
              
notes1_ptbxl=data_ptbxl['notes1']
notes2_ptbxl=data_ptbxl['notes2']
notes3_ptbxl=data_ptbxl['notes3']


# Extraer variable de sexo para cada grupo en PTB
sex_ptb_ctrl = np.array([notes1_ptb[i][0][1][1][0][0] for i in range(len(notes1_ptb))])  # CTRL
sex_ptb_mi7 = np.array([notes2_ptb[i][0][1][1][0][0] for i in range(len(notes2_ptb))])  # MI7
sex_ptb_mi60 = np.array([notes3_ptb[i][0][1][1][0][0] for i in range(len(notes3_ptb))])  # MI60


# Similarmente para PTBXL
sex_ptbxl_ctrl = np.array([notes1_ptbxl[i][0][1][0][3][0][0] for i in range(len(notes1_ptbxl))])  # CTRL
sex_ptbxl_mi7 = np.array([notes2_ptbxl[i][0][1][0][3][0][0] for i in range(len(notes2_ptbxl))])  # MI7
sex_ptbxl_mi60 = np.array([notes3_ptbxl[i][0][1][0][3][0][0] for i in range(len(notes3_ptbxl))])  # MI60

# Mapear 0 -> 'male' y 1 -> 'female'
sex_ptbxl_ctrl = np.array(['male' if notes1_ptbxl[i][0][1][0][3][0][0] == 0 else 'female' for i in range(len(notes1_ptbxl))])
sex_ptbxl_mi7 = np.array(['male' if notes2_ptbxl[i][0][1][0][3][0][0] == 0 else 'female' for i in range(len(notes2_ptbxl))])
sex_ptbxl_mi60 = np.array(['male' if notes3_ptbxl[i][0][1][0][3][0][0] == 0 else 'female' for i in range(len(notes3_ptbxl))])

sexo_ctrl =np.concatenate((sex_ptb_ctrl, sex_ptbxl_ctrl), axis=0)
sexo_mi7 =np.concatenate((sex_ptb_mi7, sex_ptbxl_mi7), axis=0)
sexo_mi60 =np.concatenate((sex_ptb_mi60, sex_ptbxl_mi60), axis=0)


# Extraer variable de edad para cada grupo en PTB
edad_ptb_ctrl = np.array([notes1_ptb[i][0][1][0][0][0] for i in range(len(notes1_ptb))])  # CTRL
edad_ptb_mi7 = np.array([notes2_ptb[i][0][1][0][0][0] for i in range(len(notes2_ptb))])  # MI7
edad_ptb_mi60 = np.array([notes3_ptb[i][0][1][0][0][0] for i in range(len(notes3_ptb))])  # MI60

# Similarmente para PTBXL
edad_ptbxl_ctrl = np.array([notes1_ptbxl[i][0][1][0][2][0][0] for i in range(len(notes1_ptbxl))])  # CTRL
edad_ptbxl_mi7 = np.array([notes2_ptbxl[i][0][1][0][2][0][0] for i in range(len(notes2_ptbxl))])  # MI7
edad_ptbxl_mi60 = np.array([notes3_ptbxl[i][0][1][0][2][0][0] for i in range(len(notes3_ptbxl))])  # MI60

# Concatenar los arreglos
edad_ctrl = np.concatenate((edad_ptb_ctrl, edad_ptbxl_ctrl), axis=0)
edad_mi7 = np.concatenate((edad_ptb_mi7, edad_ptbxl_mi7), axis=0)
edad_mi60 = np.concatenate((edad_ptb_mi60, edad_ptbxl_mi60), axis=0)

# Asegurarse de que los arreglos sean de tipo 'object' para manejar tanto números como cadenas
edad_ctrl = np.array(edad_ctrl, dtype=object)  # Convierte a tipo object
edad_mi7 = np.array(edad_mi7, dtype=object)    # Convierte a tipo object
edad_mi60 = np.array(edad_mi60, dtype=object)  # Convierte a tipo object

# Reemplazar valores no numéricos con NaN
edad_ctrl = np.where(np.isin(edad_ctrl, ['n/a', 'NaN', '']), np.nan, edad_ctrl)
edad_mi7 = np.where(np.isin(edad_mi7, ['n/a', 'NaN', '']), np.nan, edad_mi7)
edad_mi60 = np.where(np.isin(edad_mi60, ['n/a', 'NaN', '']), np.nan, edad_mi60)

# Convertir los arreglos a tipo float, ya que ahora los valores no numéricos fueron reemplazados con NaN
edad_ctrl = edad_ctrl.astype(float)
edad_mi7 = edad_mi7.astype(float)
edad_mi60 = edad_mi60.astype(float)

# Filtrar valores numéricos (eliminar NaN)
edad_ctrl = edad_ctrl[~np.isnan(edad_ctrl)]
edad_mi7 = edad_mi7[~np.isnan(edad_mi7)]
edad_mi60 = edad_mi60[~np.isnan(edad_mi60)]



rmsnoisegrupo1 = np.concatenate((data_ptb['rmsnoisegrupo1'], data_ptbxl['rmsnoisegrupo1']), axis=0)
rmsnoisegrupo2 = np.concatenate((data_ptb['rmsnoisegrupo2'], data_ptbxl['rmsnoisegrupo2']), axis=0)
rmsnoisegrupo3 = np.concatenate((data_ptb['rmsnoisegrupo3'], data_ptbxl['rmsnoisegrupo3']), axis=0)

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

Semp1 = np.full((2, NLEADS), np.nan)
Semp2 = np.full((2, NLEADS), np.nan)
Semp3 = np.full((2, NLEADS), np.nan)

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

# Variables para almacenar resultados de la validación cruzada
# Inicializar listas para almacenar resultados
CURVAS_ROC = {lead: {'individual': [], 'combined': []} for lead in range(13)}

group_labels = ['CTL', 'MI7', 'MI60']  # Define los grupos


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
    
    mg1[i] = np.nanmean(g1)  # CALCULO LA MEDIA Y SEM
    mg2[i] = np.nanmean(g2)
    mg3[i] = np.nanmean(g3)
    mc1[i] = np.nanmean(c1)
    mc2[i] = np.nanmean(c2)  
    mc3[i] = np.nanmean(c3)    
    
    #Semg1 = np.nanstd(g1)*np.array([[1], [1]])
    #Semg2 = np.nanstd(g2)*np.array([[1], [1]])
    #Semg3 = np.nanstd(g3)*np.array([[1], [1]])
    #Semc1 = np.nanstd(c1)*np.array([[1], [1]])
    #Semc2 = np.nanstd(c2)* np.array([[1], [1]])
    #Semc3 = np.nanstd(c3)* np.array([[1], [1]])    
    
 
    Semg1[:, i] = np.array([1, 1]) * np.std(g1, ddof=0)  # ddof=0 para el denominador N
    Semg2[:, i] = np.array([1, 1]) * np.std(g2, ddof=0)
    Semg3[:, i] = np.array([1, 1]) * np.std(g3, ddof=0)

    Semc1[:, i] = np.array([1, 1]) * np.std(c1, ddof=0)
    Semc2[:, i] = np.array([1, 1]) * np.std(c2, ddof=0)
    Semc3[:, i] = np.array([1, 1]) * np.std(c3, ddof=0)
    

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

    # A PARTIR DE ACÁ HAGO VALIDACIÓN CRUZADA--------------------
        # Validación cruzada para índices individuales
 
    
    pairs_individuales = [
        (g1, g2),   # entropias
        (g1, g3),
        (g2, g3),
        (c1, c2),   # complejidades  
        (c1, c3),
        (c2, c3),
        (edad_ctrl,edad_mi7),
        (edad_ctrl,edad_mi60),
        (edad_mi7,edad_mi60),
    ]

    stat, p_value = stats.kruskal(g1, g2, g3)
    print(f'Prueba de Kruskal-Wallis - Estadístico-entropia: {stat}, p-valor: {p_value}')
    
    stat, p_value = stats.kruskal(c1, c2, c3)
    print(f'Prueba de Kruskal-Wallis - Estadístico-complejidad: {stat}, p-valor: {p_value}')
    
    
    for idx1, idx2 in pairs_individuales:
        labels = np.array([1] * len(idx1) + [0] * len(idx2))
        features = np.concatenate([idx1, idx2]).reshape(-1, 1)
        # Suponiendo que tienes un array `patient_ids` que contiene los IDs de los pacientes
        resultados = cross_validate(features, labels, todos_los_grupos)
        # Calcular el p-value
        # Calcular el p-value usando la prueba de Mann-Whitney U
        #Si las muestras de los grupos son de distinto tamaño, deberías 
        #considerar una prueba estadística que no requiera que las muestras 
        #sean del mismo tamaño, como la prueba de Mann-Whitney U. Esta prueba es adecuada para comparar dos grupos independientes, sin necesidad de emparejarlos.
        stat, p_valor = mannwhitneyu(idx1, idx2, alternative='two-sided')
      
 
        # Identificar el grupo y el tipo de índice
        if (idx1 is g1 or idx1 is g2 or idx1 is g3) and (idx2 is g1 or idx2 is g2 or idx2 is g3):  # Entropía
           group_name = f"{group_labels[0]}-{group_labels[1]}" if idx1 is g1 and idx2 is g2 else \
                     f"{group_labels[0]}-{group_labels[2]}" if idx1 is g1 and idx2 is g3 else \
                     f"{group_labels[1]}-{group_labels[2]}"
           index_type = 'Entropía'
        elif (idx1 is c1 or idx1 is c2 or idx1 is c3) and (idx2 is c1 or idx2 is c2 or idx2 is c3):  # Complejidad
           group_name = f"{group_labels[0]}-{group_labels[1]}" if idx1 is c1 and idx2 is c2 else \
                     f"{group_labels[0]}-{group_labels[2]}" if idx1 is c1 and idx2 is c3 else \
                     f"{group_labels[1]}-{group_labels[2]}"
           index_type = 'Complejidad'
        else:
           group_name = f"{group_labels[0]}-{group_labels[1]}" if idx1 is edad_ctrl and idx2 is edad_mi7 else \
                     f"{group_labels[0]}-{group_labels[2]}" if idx1 is edad_ctrl and idx2 is edad_mi60 else \
                     f"{group_labels[1]}-{group_labels[2]}"
           index_type = 'Edad'
 
           

        CURVAS_ROC[i]['individual'].append({
        'sens_mean': resultados['sens_mean'],
        'sens_min': resultados['sens_min'],
        'sens_std': resultados['sens_std'],
        'spec_mean': resultados['spec_mean'],
        'spec_min': resultados['spec_min'],
        'spec_std': resultados['spec_std'],
        'auc_mean': resultados['auc_mean'],
        'auc_min': resultados['auc_min'],
        'auc_std': resultados['auc_std'],
        'p_valor': p_valor,
        'Grupos': f"{group_name} ({index_type})"
    })

        
    # Validación cruzada para índices conjuntos
    conjuntos = [
        (np.vstack((g1, c1)).T, np.vstack((g2, c2)).T),
        (np.vstack((g1, c1)).T, np.vstack((g3, c3)).T),
        (np.vstack((g2, c2)).T, np.vstack((g3, c3)).T)
    ]
    
    for features1, features2 in conjuntos:
        labels = np.array([1] * len(features1) + [0] * len(features2))
        features = np.vstack((features1, features2))
        # Suponiendo que tienes un array `patient_ids` que contiene los IDs de los pacientes
        resultados = cross_validate(features, labels, todos_los_grupos)

        # Inicializar combined_group_name
        combined_group_name = ""

        # Identificar los grupos para conjuntos
        if np.array_equal(features1, np.vstack((g1, c1)).T) and np.array_equal(features2, np.vstack((g2, c2)).T):
           combined_group_name = "CTRL-MI7"
        elif np.array_equal(features1, np.vstack((g1, c1)).T) and np.array_equal(features2, np.vstack((g3, c3)).T):
           combined_group_name = "CTRL-MI60"
        elif np.array_equal(features1, np.vstack((g2, c2)).T) and np.array_equal(features2, np.vstack((g3, c3)).T):
           combined_group_name = "MI7-MI60"
    
        # Verificar que combined_group_name no esté vacío
        if combined_group_name:  # Solo agregar si se definió correctamente
          # Establecer el tipo de índice
          index_type = "Ent-Comp"


        
        CURVAS_ROC[i]['combined'].append({
        'sens_mean': resultados['sens_mean'],
        'sens_min': resultados['sens_min'],
        'sens_std': resultados['sens_std'],
        'spec_mean': resultados['spec_mean'],
        'spec_min': resultados['spec_min'],
        'spec_std': resultados['spec_std'],
        'auc_mean': resultados['auc_mean'],
        'auc_min': resultados['auc_min'],
        'auc_std': resultados['auc_std'],
        'p_valor': p_valor,
        'Grupos': f"{combined_group_name} ({index_type})"
    })
        
    # Asegúrate de tener las funciones necesarias para realizar las permutaciones y calcular las estadísticas.
# (Asegúrate de que las funciones permute_and_calc_stat y calc_observed_stat estén definidas correctamente.)

    n_permutations = 1000

# Definir los grupos
    grupo_A = pd.DataFrame({'Ent': g1, 'Comp': c1, 'Grupo': ['A']*len(g1)})
    grupo_B = pd.DataFrame({'Ent': g2, 'Comp': c2, 'Grupo': ['B']*len(g2)})
    grupo_C = pd.DataFrame({'Ent': g3, 'Comp': c3, 'Grupo': ['C']*len(g3)})

# Concatenar los tres grupos
    data_all = pd.concat([grupo_A, grupo_B, grupo_C], axis=0)

# Etiquetas de los grupos
    labels = np.array([0] * len(grupo_A) + [1] * len(grupo_B) + [2] * len(grupo_C))

# Realizar las permutaciones
    permuted_stats = permute_and_calc_stat(data_all, labels, n_permutations)

# Calcular la estadística observada
    observed_stat = calc_observed_stat(data_all, labels)

# Calcular el valor p del test de permutación multivariante
    p_value = np.sum(permuted_stats >= observed_stat) / n_permutations

# Resultados iniciales
    print(f"Estadística observada en {leadNames[i]}: {observed_stat:} ")
    print(f"Valor p del test de permutación multivariante en {leadNames[i]}: {p_value:.10f}")

# Si el valor p es menor que 0.05, realizamos comparaciones post-hoc
    if p_value < 0.05:
    # Ajuste Bonferroni para las comparaciones
       group_combinations = list(combinations(['A', 'B', 'C'], 2))  # Generar las combinaciones de grupos
       bonferroni_alpha = 0.05 / len(group_combinations)  # Ajuste Bonferroni por el número de combinaciones

       print(f"\nAjuste Bonferroni: {bonferroni_alpha:}")
    
    # Comparar por pares
       for group_1, group_2 in group_combinations:
           observed_stat, p_value = compare_groups_posthoc(data_all, group_1, group_2, n_permutations)
           print(f"\nComparación entre {group_1} y {group_2} en {leadNames[i]}:")
           print(f"  Estadística observada en {leadNames[i]}: {observed_stat:.10f}")
           print(f"  Valor p en {leadNames[i]}: {p_value:.10f}")
        
           if p_value < bonferroni_alpha:
              print(f"  Hay diferencia estadísticamente significativa entre {group_1} y {group_2}")
           else:
              print(f"  No hay diferencia estadísticamente significativa entre {group_1} y {group_2}")

#------------------------------------------------------------------

plt.figure() 
# con density true se normaliza el histograma
plt.hist(edad_ctrl, bins=15, density=True, alpha=0.7, color='green', edgecolor='black')       
plt.hist(edad_mi7, bins=15, density=True, alpha=0.7, color='red', edgecolor='black') 
plt.hist(edad_mi60, bins=15, density=True, alpha=0.7, color='yellow', edgecolor='black')
# Añadir una línea vertical para la media
plt.axvline(np.median(edad_ctrl), color='green', linestyle='dashed', linewidth=2)
plt.axvline(np.median(edad_mi7), color='red', linestyle='dashed', linewidth=2)
plt.axvline(np.median(edad_mi60), color='yellow', linestyle='dashed', linewidth=2)
#como deberian ser los histogramas de dos muestras que tienen p valor menor q 0.05?
# Densidades Diferentes: Si se normalizan los histogramas (como se mencionó anteriormente), 
# las alturas de las barras en el histograma podrían mostrar que una 
# muestra tiene una mayor densidad en ciertos rangos de valores en 
# comparación con la otra


# Contar la cantidad de hombres y mujeres en cada grupo
sexo_count_ctrl = pd.Series(sexo_ctrl).value_counts()
sexo_count_mi7 = pd.Series(sexo_mi7).value_counts()
sexo_count_mi60 = pd.Series(sexo_mi60).value_counts()

# Asegurarse de que todos los grupos tengan los mismos niveles de sexo (male, female)
sexo_count_ctrl = sexo_count_ctrl.reindex(['male', 'female'], fill_value=0)
sexo_count_mi7 = sexo_count_mi7.reindex(['male', 'female'], fill_value=0)
sexo_count_mi60 = sexo_count_mi60.reindex(['male', 'female'], fill_value=0)

# Crear un DataFrame para organizar los datos
df_sexos = pd.DataFrame({
    'Ctrl': sexo_count_ctrl,
    'Mi7': sexo_count_mi7,
    'Mi60': sexo_count_mi60
})


# Crear el gráfico de barras
df_sexos.plot(kind='bar', stacked=False, figsize=(10, 6))

# Personalizar el gráfico
plt.title('Distribución de Sexo por Grupo')
plt.xlabel('Sexo')
plt.ylabel('Frecuencia')
plt.xticks(rotation=0)  # Etiquetas de sexo en horizontal
plt.legend(title='Grupo', labels=['Ctrl', 'Mi7', 'Mi60'])
plt.tight_layout()

# Mostrar el gráfico
plt.show()


plt.figure() 
# con density true se normaliza el histograma
plt.hist(sexo_ctrl, bins=15, density=True, alpha=0.7, color='green', edgecolor='black')       
plt.hist(sexo_mi7, bins=15, density=True, alpha=0.7, color='red', edgecolor='black') 
plt.hist(sexo_mi60, bins=15, density=True, alpha=0.7, color='yellow', edgecolor='black')
# Mostrar el gráfico
plt.show()

#------------------------------------------------------
#---- EDAD
#------------------------------------------------------
# 1. **ANÁLISIS DE NORMALIDAD** (Shapiro-Wilk) para Edad
# ------------------------------------------------------
# PTB
# Realizamos el test de normalidad (Shapiro-Wilk) para cada grupo.
# Si el p-valor es menor que 0.05, rechazamos la hipótesis nula de normalidad.

normalidad_ctrl_ptb = shapiro(edad_ptb_ctrl) 
normalidad_mi7_ptb= shapiro(edad_ptb_mi7)
normalidad_mi60_ptb = shapiro(edad_ptb_mi60)

print("Test de normalidad (edad PTB):")
print(f"CTRL (edad PTB): p = {normalidad_ctrl_ptb.pvalue}")
print(f"MI7 (edad PTB): p = {normalidad_mi7_ptb.pvalue}")
print(f"MI60 (edad PTB: p = {normalidad_mi60_ptb.pvalue}")

# Si alguno de los grupos no es normal, se recomendaría usar pruebas no paramétricas.

# 
# 2. **Prueba ANOVA** (paramétrica) si los datos son normales
# 

# Si los datos son normales, realizamos ANOVA para comparar los tres grupos.
if normalidad_ctrl_ptb.pvalue > 0.05 and normalidad_mi7_ptb.pvalue > 0.05 and normalidad_mi60_ptb.pvalue > 0.05:
    # ANOVA para comparar los tres grupos
    stat_ptb, p_value_ptb = f_oneway(edad_ptb_ctrl, edad_ptb_mi7, edad_ptb_mi60)
    print("\nANOVA:")
    print(f"p = {p_value_ptb}")

    # Si el p-valor es menor que 0.05, hay diferencias significativas
    if p_value_ptb < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc en PTB.")
        
        # Comparaciones post-hoc usando el test de Tukey
        all_ages_ptb = np.concatenate([edad_ptb_ctrl, edad_ptb_mi7, edad_ptb_mi60])
        groups_ptb = ['CTRL'] * len(edad_ptb_ctrl) + ['MI7'] * len(edad_ptb_mi7) + ['MI60'] * len(edad_ptb_mi60)
        tukey_result_ptb = pairwise_tukeyhsd(all_ages_ptb, groups_ptb)
        print(tukey_result_ptb)
        
else:
    print("\nLos datos no son normales. Realizando prueba de Kruskal-Wallis (no paramétrica) en PTB.")

# 
# 3. **Prueba de Kruskal-Wallis** (si no son normales)
# 

# Si los datos no son normales, usamos Kruskal-Wallis (prueba no paramétrica)
stat_ptb, p_value_kw_ptb = kruskal(edad_ptb_ctrl, edad_ptb_mi7, edad_ptb_mi60)
print("\nKruskal-Wallis en PTB:")
print(f"p = {p_value_kw_ptb}")

# Si el p-valor es menor que 0.05, hay diferencias significativas
if p_value_kw_ptb < 0.05:
    print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc en PTB.")
    
    # Comparaciones por pares usando Mann-Whitney U
    stat_ptb, p_ctrl_mi7_ptb = mannwhitneyu(edad_ptb_ctrl, edad_ptb_mi7)
    stat_ptb, p_ctrl_mi60_ptb = mannwhitneyu(edad_ptb_ctrl, edad_ptb_mi60)
    stat_ptb, p_mi7_mi60_ptb = mannwhitneyu(edad_ptb_mi7, edad_ptb_mi60)

    print(f"CTRL vs MI7, PTB: p = {p_ctrl_mi7_ptb}")
    print(f"CTRL vs MI60, PTB: p = {p_ctrl_mi60_ptb}")
    print(f"MI7 vs MI60, PTB: p = {p_mi7_mi60_ptb}")


#------------------------------------------------------
# 1. **ANÁLISIS DE NORMALIDAD** (Shapiro-Wilk) para Edad
# ------------------------------------------------------
# PTB XL
# Realizamos el test de normalidad (Shapiro-Wilk) para cada grupo.
# Si el p-valor es menor que 0.05, rechazamos la hipótesis nula de normalidad.

normalidad_ctrl_ptbxl = shapiro(edad_ptbxl_ctrl) 
normalidad_mi7_ptbxl= shapiro(edad_ptbxl_mi7)
normalidad_mi60_ptbxl = shapiro(edad_ptbxl_mi60)

print("Test de normalidad (edad PTBXL):")
print(f"CTRL (edad PTBXL): p = {normalidad_ctrl_ptbxl.pvalue}")
print(f"MI7 (edad PTBXL): p = {normalidad_mi7_ptbxl.pvalue}")
print(f"MI60 (edad PTBXL: p = {normalidad_mi60_ptbxl.pvalue}")

# Si alguno de los grupos no es normal, se recomendaría usar pruebas no paramétricas.

# 
# 2. **Prueba ANOVA** (paramétrica) si los datos son normales
# 

# Si los datos son normales, realizamos ANOVA para comparar los tres grupos.
if normalidad_ctrl_ptbxl.pvalue > 0.05 and normalidad_mi7_ptbxl.pvalue > 0.05 and normalidad_mi60_ptbxl.pvalue > 0.05:
    # ANOVA para comparar los tres grupos
    stat_ptbxl, p_value_ptbxl = f_oneway(edad_ptbxl_ctrl, edad_ptbxl_mi7, edad_ptbxl_mi60)
    print("\nANOVA:")
    print(f"p = {p_value_ptbxl}")

    # Si el p-valor es menor que 0.05, hay diferencias significativas
    if p_value_ptbxl < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc en PTBXL.")
        
        # Comparaciones post-hoc usando el test de Tukey
        all_ages_ptbxl = np.concatenate([edad_ptbxl_ctrl, edad_ptbxl_mi7, edad_ptbxl_mi60])
        groups_ptbxl = ['CTRL'] * len(edad_ptbxl_ctrl) + ['MI7'] * len(edad_ptbxl_mi7) + ['MI60'] * len(edad_ptbxl_mi60)
        tukey_result_ptbxl = pairwise_tukeyhsd(all_ages_ptbxl, groups_ptbxl)
        print(tukey_result_ptbxl)
        
else:
    print("\nLos datos no son normales. Realizando prueba de Kruskal-Wallis (no paramétrica) en PTBXL.")

# 
# 3. **Prueba de Kruskal-Wallis** (si no son normales)
# 

# Si los datos no son normales, usamos Kruskal-Wallis (prueba no paramétrica)
stat_ptbxl, p_value_kw_ptbxl = kruskal(edad_ptbxl_ctrl, edad_ptbxl_mi7, edad_ptbxl_mi60)
print("\nKruskal-Wallis en PTBXL:")
print(f"p = {p_value_kw_ptbxl}")

# Si el p-valor es menor que 0.05, hay diferencias significativas
if p_value_kw_ptbxl < 0.05:
    print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc en PTBXL.")
    
    # Comparaciones por pares usando Mann-Whitney U
    stat_ptbxl, p_ctrl_mi7_ptbxl = mannwhitneyu(edad_ptbxl_ctrl, edad_ptbxl_mi7)
    stat_ptbxl, p_ctrl_mi60_ptbxl = mannwhitneyu(edad_ptbxl_ctrl, edad_ptbxl_mi60)
    stat_ptbxl, p_mi7_mi60_ptbxl = mannwhitneyu(edad_ptbxl_mi7, edad_ptbxl_mi60)

    print(f"CTRL vs MI7, PTB XL: p = {p_ctrl_mi7_ptbxl}")
    print(f"CTRL vs MI60, PTB XL: p = {p_ctrl_mi60_ptbxl}")
    print(f"MI7 vs MI60, PTB XL: p = {p_mi7_mi60_ptbxl}")





# ------------------------------------------------------
# 1. **ANÁLISIS DE NORMALIDAD** (Shapiro-Wilk) para Edad
# 
# PTB + PTB XL
# Realizamos el test de normalidad (Shapiro-Wilk) para cada grupo.
# Si el p-valor es menor que 0.05, rechazamos la hipótesis nula de normalidad.

normalidad_ctrl = shapiro(edad_ctrl)
normalidad_mi7 = shapiro(edad_mi7)
normalidad_mi60 = shapiro(edad_mi60)

print("Test de normalidad (edad):")
print(f"CTRL (edad): p = {normalidad_ctrl.pvalue}")
print(f"MI7 (edad): p = {normalidad_mi7.pvalue}")
print(f"MI60 (edad): p = {normalidad_mi60.pvalue}")

# Si alguno de los grupos no es normal, se recomendaría usar pruebas no paramétricas.

# 
# 2. **Prueba ANOVA** (paramétrica) si los datos son normales
# 

# Si los datos son normales, realizamos ANOVA para comparar los tres grupos.
if normalidad_ctrl.pvalue > 0.05 and normalidad_mi7.pvalue > 0.05 and normalidad_mi60.pvalue > 0.05:
    # ANOVA para comparar los tres grupos
    stat, p_value = f_oneway(edad_ctrl, edad_mi7, edad_mi60)
    print("\nANOVA:")
    print(f"p = {p_value}")

    # Si el p-valor es menor que 0.05, hay diferencias significativas
    if p_value < 0.05:
        print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc.")
        
        # Comparaciones post-hoc usando el test de Tukey
        all_ages = np.concatenate([edad_ctrl, edad_mi7, edad_mi60])
        groups = ['CTRL'] * len(edad_ctrl) + ['MI7'] * len(edad_mi7) + ['MI60'] * len(edad_mi60)
        tukey_result = pairwise_tukeyhsd(all_ages, groups)
        print(tukey_result)
        
else:
    print("\nLos datos no son normales. Realizando prueba de Kruskal-Wallis (no paramétrica).")

#
# 3. **Prueba de Kruskal-Wallis** (si no son normales)
# 

# Si los datos no son normales, usamos Kruskal-Wallis (prueba no paramétrica)
stat, p_value_kw = kruskal(edad_ctrl, edad_mi7, edad_mi60)
print("\nKruskal-Wallis:")
print(f"p = {p_value_kw}")

# Si el p-valor es menor que 0.05, hay diferencias significativas
if p_value_kw < 0.05:
    print("Hay diferencias significativas entre al menos dos grupos, realizaremos comparaciones post-hoc.")
    
    # Comparaciones por pares usando Mann-Whitney U
    stat, p_ctrl_mi7 = mannwhitneyu(edad_ctrl, edad_mi7)
    stat, p_ctrl_mi60 = mannwhitneyu(edad_ctrl, edad_mi60)
    stat, p_mi7_mi60 = mannwhitneyu(edad_mi7, edad_mi60)

    print(f"CTRL vs MI7, PTB+PTBXL: p = {p_ctrl_mi7}")
    print(f"CTRL vs MI60, PTB+PTBXL: p = {p_ctrl_mi60}")
    print(f"MI7 vs MI60, PTB+PTBXL: p = {p_mi7_mi60}")

# ----------------------------------------------------------------------
# 4. **Comparaciones de Sexo**: Chi-cuadrado para variables categóricas
# 
# PTB
sexo_ctrl_ptb_count = pd.Series(sex_ptb_ctrl).value_counts()
sexo_mi7_ptb_count = pd.Series(sex_ptb_mi7).value_counts()
sexo_mi60_ptb_count = pd.Series(sex_ptb_mi60).value_counts()

# Crear una tabla de contingencia (crosstab)
sexo_table_ptb = pd.DataFrame({
    'CTRL': sexo_ctrl_ptb_count,
    'MI7': sexo_mi7_ptb_count,
    'MI60': sexo_mi60_ptb_count
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat_ptb, p_value_sexo_ptb, dof_ptb, expected_ptb = chi2_contingency(sexo_table_ptb)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo_ptb}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo_ptb < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre los grupos de la PTB.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre los grupos de la PTB.")


# PTB XL
sexo_ctrl_ptbxl_count = pd.Series(sex_ptbxl_ctrl).value_counts()
sexo_mi7_ptbxl_count = pd.Series(sex_ptbxl_mi7).value_counts()
sexo_mi60_ptbxl_count = pd.Series(sex_ptbxl_mi60).value_counts()

# Crear una tabla de contingencia (crosstab)
sexo_table_ptbxl = pd.DataFrame({
    'CTRL': sexo_ctrl_ptbxl_count,
    'MI7': sexo_mi7_ptbxl_count,
    'MI60': sexo_mi60_ptbxl_count
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat_ptbxl, p_value_sexo_ptbxl, dof_ptbxl, expected_ptbxl = chi2_contingency(sexo_table_ptbxl)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo_ptbxl}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo_ptbxl < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre los grupos de la PTB XL.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre los grupos de la PTB XL.")


# Suma de PTB + PTB XL
# Contar las frecuencias de sexo en cada grupo
sexo_ctrl_count = pd.Series(sexo_ctrl).value_counts()
sexo_mi7_count = pd.Series(sexo_mi7).value_counts()
sexo_mi60_count = pd.Series(sexo_mi60).value_counts()

# Crear una tabla de contingencia (crosstab)
sexo_table = pd.DataFrame({
    'CTRL': sexo_ctrl_count,
    'MI7': sexo_mi7_count,
    'MI60': sexo_mi60_count
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat, p_value_sexo, dof, expected = chi2_contingency(sexo_table)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre los grupos.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre los grupos.")


# CTRL vs MI7

# Crear una tabla de contingencia (crosstab)
sexo_table = pd.DataFrame({
    'CTRL': sexo_ctrl_count,
    'MI7': sexo_mi7_count,
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat, p_value_sexo, dof, expected = chi2_contingency(sexo_table)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre CTRL y MI7.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre CTRL y MI7.")

# CTRL y MI60


# Crear una tabla de contingencia (crosstab)
sexo_table = pd.DataFrame({
    'CTRL': sexo_ctrl_count,
    'MI60': sexo_mi60_count,
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat, p_value_sexo, dof, expected = chi2_contingency(sexo_table)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre CTRL y MI60.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre CTRL y MI60.")


# MI7 y MI60


# Crear una tabla de contingencia (crosstab)
sexo_table = pd.DataFrame({
    'MI7': sexo_mi7_count,
    'MI60': sexo_mi60_count,
}).fillna(0)  # Rellenamos con ceros si algún grupo no tiene esa categoría

# Test de Chi-cuadrado
chi2_stat, p_value_sexo, dof, expected = chi2_contingency(sexo_table)
print("\nTest de Chi-cuadrado para Sexo:")
print(f"p = {p_value_sexo}")

# Si el p-valor es menor que 0.05, hay diferencias significativas en la distribución de sexo.
if p_value_sexo < 0.05:
    print("Hay diferencias significativas en la distribución de sexo entre MI7 y MI60.")
else:
    print("No hay diferencias significativas en la distribución de sexo entre MI7 y MI60.")


# ------------------------------------------------------
# 5. **Análisis de poder estadístico** (Power Analysis)
# ------------------------------------------------------

power_analysis = TTestIndPower()

# El tamaño de la muestra del primer grupo (CTRL)
nobs1 = len(edad_ctrl)

# El tamaño de la muestra del segundo grupo (MI7)
nobs2 = len(edad_mi7)

# El tamaño de la muestra del segundo grupo (MI60)
nobs3=len(edad_mi60)

# Calcular la relación entre los tamaños de muestra
ratio_mi7_ctrl = nobs2 / nobs1

ratio_mi60_ctrl = nobs3 / nobs1

ratio_mi60_mi7 = nobs3 / nobs2


# Realizar el análisis de poder
power_mi7_ctrl= power_analysis.solve_power(effect_size=0.5, nobs1=nobs1, alpha=0.05, ratio=ratio_mi7_ctrl)

power_mi60_ctrl = power_analysis.solve_power(effect_size=0.5, nobs1=nobs1, alpha=0.05, ratio=ratio_mi60_ctrl)

power_mi60_mi7 = power_analysis.solve_power(effect_size=0.5, nobs1=nobs2, alpha=0.05, ratio=ratio_mi60_mi7)

# Mostrar el resultado
print("\nAnálisis de poder estadístico:")
print(f"Poder estadístico entre CTRL y MI7: {power_mi7_ctrl}")
print(f"Poder estadístico entre CTRL y MI60: {power_mi60_ctrl}")
print(f"Poder estadístico entre MI7 y MI60: {power_mi60_mi7}")
# Si el poder es menor que 0.8, podrías necesitar aumentar el tamaño de la muestra.


# --------------------------------------------------------
# Cálculo de media y desvío estandar de la edad
# --------------------------------------------------------
import numpy as np
import pandas as pd

# Convertir a numérico y manejar errores como NaN
edad_ptb_ctrl = pd.to_numeric(edad_ptb_ctrl, errors='coerce')
edad_ptb_mi7 = pd.to_numeric(edad_ptb_mi7, errors='coerce')
edad_ptb_mi60 = pd.to_numeric(edad_ptb_mi60, errors='coerce')

edad_ptbxl_ctrl = pd.to_numeric(edad_ptbxl_ctrl, errors='coerce')
edad_ptbxl_mi7 = pd.to_numeric(edad_ptbxl_mi7, errors='coerce')
edad_ptbxl_mi60 = pd.to_numeric(edad_ptbxl_mi60, errors='coerce')

# Eliminar NaN usando np.isnan() y indexado booleano
edad_ptb_ctrl = edad_ptb_ctrl[~np.isnan(edad_ptb_ctrl)]
edad_ptb_mi7 = edad_ptb_mi7[~np.isnan(edad_ptb_mi7)]
edad_ptb_mi60 = edad_ptb_mi60[~np.isnan(edad_ptb_mi60)]

edad_ptbxl_ctrl = edad_ptbxl_ctrl[~np.isnan(edad_ptbxl_ctrl)]
edad_ptbxl_mi7 = edad_ptbxl_mi7[~np.isnan(edad_ptbxl_mi7)]
edad_ptbxl_mi60 = edad_ptbxl_mi60[~np.isnan(edad_ptbxl_mi60)]

# Función para calcular mediana y rango intercuartílico (IQR)
def calcular_mediana_iqr(arr):
    mediana = np.median(arr)
    q25 = np.percentile(arr, 25)  # Percentil 25
    q75 = np.percentile(arr, 75)  # Percentil 75
    iqr = q75 - q25
    return f"{mediana:.2f} (IQR: {q25:.2f} - {q75:.2f})"



# Calcular para PTBXL
mediana_ctrl_ptbxl = calcular_mediana_iqr(edad_ptbxl_ctrl)
mediana_mi7_ptbxl = calcular_mediana_iqr(edad_ptbxl_mi7)
mediana_mi60_ptbxl = calcular_mediana_iqr(edad_ptbxl_mi60)

# Crear un DataFrame para mostrar los resultados
data = {
    "Grupo": ["CTRL", "MI7", "MI60"],
    "Mediana ± IQR PTBXL": [mediana_ctrl_ptbxl, mediana_mi7_ptbxl, mediana_mi60_ptbxl]
}

df = pd.DataFrame(data)

# Imprimir la tabla
print(df)


#----------------------------------------
# Grafico plano Entropía-Complejidad
#----------------------------------------

wNscales = 16

if graficar == 1:
    # Grafico los resultados de entropía-complejidad
    if i == leads[-1]:
          plt.figure()
          ax = plt.gca()
          fsize = 14
          wcolor = 'white'
          lgcolor = 'grey'
          gcolor = 'lightgrey'
          hverrorbar(mg1[i], mc1[i], Semg1[:, i].reshape(-1, 1), Semc1[:, i].reshape(-1, 1), color='k', linestyle='-', linewidth=1)
          hverrorbar(mg2[i], mc2[i], Semg2[:, i].reshape(-1, 1), Semc2[:, i].reshape(-1, 1), color='k', linestyle='-', linewidth=1)
          hverrorbar(mg3[i], mc3[i], Semg3[:, i].reshape(-1, 1), Semc3[:, i].reshape(-1, 1), color='k', linestyle='-', linewidth=1)
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

          ax.set_xlim(0.6, 1)
          ax.set_ylim(0, 0.25)
          ax.set_xticks(np.arange(0.6, 1, 0.05))
          ax.set_yticks(np.arange(0, 0.25, 0.05))
          ax.legend(loc='upper right', fontsize=fsize)
          #fig.canvas.manager.set_window_title(strcualfiltrar[cualfiltrar])
          plt.grid(False)
          plt.show()
          
    
        
#----------------------------------------
# Grafico boxplot de Entropía
#----------------------------------------


# Si graficar es 1, se genera el gráfico
if graficar == 1:
    fsize = 12
    scnsize = [1920, 1080]  # Tamaño de la pantalla, ajusta según tu pantalla
    posscreen = [0.1 * scnsize[0], 0.25 * scnsize[1], 0.75 * scnsize[0], 0.25 * scnsize[1]]
    position = [0.08, 0.1, 0.9, 0.8]

    # Aquí se crean las figuras y ejes
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    # Suponiendo que 'leads', 'mg1', 'mg2', 'mg3', 'Semg1', 'Semg2', 'Semg3', 'sigg12', 'sigg13', 'sigg23', 'leadNames' están definidos en algún lado
    #T = leads[:-1]  # Asegúrate de que 'leads' está definido y es un arreglo NumPy o lista
    T = np.array(leads[:-1]) 
    barwidth = 1
    delta = 0.22
    colormap = plt.cm.Greys  # Usar colormap de grises

    # Graficamos los errores (con barras de error)
    eb1 = ax.errorbar(T - delta, mg1[T], yerr=[Semg1[0, T], Semg1[1, T]], fmt='ks', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='w')
    eb2 = ax.errorbar(T, mg2[T], yerr=[Semg2[0, T], Semg2[1, T]], fmt='ko', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='0.5')
    eb3 = ax.errorbar(T + delta, mg3[T], yerr=[Semg3[0, T], Semg3[1, T]], fmt='kd', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='0.75')

    # Etiquetas de texto sobre los puntos
    level12 = 0.01 * np.ones(len(T))
    level13 = 0.01 * np.ones(len(T))

    # Usar un bucle para iterar sobre T y agregar texto a cada punto
    for i in range(len(T)):
        ax.text(T[i], level12[i] + mg2[T[i]] + Semg2[1, T[i]], str(sigg12[T[i]]), fontsize=fsize, ha='center', va='bottom')
        ax.text(T[i] + 0.22, level13[i] + mg3[T[i]] + Semg3[1, T[i]], str(sigg13[T[i]]), fontsize=fsize, ha='center', va='bottom')
        ax.text(T[i] + 0.32, level13[i] + mg3[T[i]] + Semg3[1, T[i]], sigg23[T[i]], fontsize=fsize, ha='center', va='bottom')

        # Configuración de los ejes
        ax.set_xticks(T)
        ax.set_xticklabels([str(leadNames[i]) for i in T], fontsize=fsize)
        ax.set_ylabel('Normalized entropy', fontsize=fsize)
  
        # Ajustamos límites
        ax.set_ylim(0.65, 1) 
   
        # Leyenda
        ax.legend([eb1, eb2, eb3], ['CTRL', 'MI7', 'MI60'], loc='upper right')
        plt.show()


#----------------------------------------
# Grafico boxplot de Complejidad
#----------------------------------------


# Si graficar es 1, se genera el gráfico
if graficar == 1:
    fsize = 12
    scnsize = [1920, 1080]  # Tamaño de la pantalla, ajusta según tu pantalla
    posscreen = [0.1 * scnsize[0], 0.25 * scnsize[1], 0.75 * scnsize[0], 0.25 * scnsize[1]]
    position = [0.08, 0.1, 0.9, 0.8]

    # Aquí se crean las figuras y ejes
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    
    T = np.array(leads[:-1]) 
    barwidth = 1
    delta = 0.22
    colormap = plt.cm.Greys  # Usar colormap de grises

    # Graficamos los errores (con barras de error)
    eb1 = ax.errorbar(T - delta, mc1[T], yerr=[Semc1[0, T], Semc1[1, T]], fmt='ks', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='w')
    eb2 = ax.errorbar(T, mc2[T], yerr=[Semc2[0, T], Semc2[1, T]], fmt='ko', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='0.5')
    eb3 = ax.errorbar(T + delta, mc3[T], yerr=[Semc3[0, T], Semc3[1, T]], fmt='kd', linewidth=1, markersize=8, markeredgecolor='k', markerfacecolor='0.75')

    # Etiquetas de texto sobre los puntos
    level12 = 0.01 * np.ones(len(T))
    level13 = 0.01 * np.ones(len(T))

    # Usar un bucle para iterar sobre T y agregar texto a cada punto
    for i in range(len(T)):
        ax.text(T[i], level12[i] + mc2[T[i]] + Semc2[1, T[i]], str(sigc12[T[i]]), fontsize=fsize, ha='center', va='bottom')
        ax.text(T[i] + 0.22, level13[i] + mc3[T[i]] + Semc3[1, T[i]], str(sigc13[T[i]]), fontsize=fsize, ha='center', va='bottom')
        ax.text(T[i] + 0.32, level13[i] + mc3[T[i]] + Semc3[1, T[i]], sigc23[T[i]], fontsize=fsize, ha='center', va='bottom')

        # Configuración de los ejes
        ax.set_xticks(T)
        ax.set_xticklabels([str(leadNames[i]) for i in T], fontsize=fsize)
        ax.set_ylabel('Normalized complexity', fontsize=fsize)
  
        # Ajustamos límites
        ax.set_ylim(0.05, 0.3)  
   
        # Leyenda
        ax.legend([eb1, eb2, eb3], ['CTRL', 'MI7', 'MI60'], loc='lower right')
        plt.show()

         
plt.figure()
plt.plot(g1w,c1w,'*b', label='CTRL' )
plt.plot(g2w,c2w,'*r', label='MI7' )
plt.plot(g3w,c3w,'*g', label='MI60' )

# pacientes atipicos: 
indices_cercanos = np.array([1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1483, 1703, 1703,
                             1703, 1703, 1703, 1703, 1703, 1703, 1703, 1703, 1703, 1703])

# Usar los índices para encontrar los pacientes correspondientes
pacientes_correspondientes = grupo1Name[indices_cercanos]

# Eliminar duplicados si hay índices repetidos
pacientes_correspondientes_unicos = np.unique(pacientes_correspondientes)