import pickle
import scipy.io
from scipy.io import savemat

# Base de datos PTB
# Health control
PTBHE = [
    'patient104', 'patient105', 'patient116', 'patient117', 'patient121',
    'patient122', 'patient131', 'patient150', 'patient155', 'patient165', 'patient166',
    'patient169', 'patient170', 'patient172', 'patient173', 'patient174', 'patient180e',
    'patient182', 'patient185', 'patient198b', 'patient214', 'patient229b', 'patient233',
    'patient234', 'patient235', 'patient236', 'patient237', 'patient238', 'patient239',
    'patient240', 'patient241b', 'patient243', 'patient244', 'patient246',
    'patient247', 'patient248', 'patient251c', 'patient252', 'patient255', 'patient260',
    'patient263', 'patient264', 'patient266', 'patient267', 'patient276', 'patient277',
    'patient279d', 'patient284c'
]

# ecg < 7 días del infarto (healing) SIN VT/VF
PTBMI07NOVTVF = [
    'patient001', 'patient004', 'patient005', 'patient006', 'patient007', 'patient008', 'patient010', 'patient011',
    'patient012', 'patient013', 'patient014', 'patient015', 'patient016', 'patient017', 'patient018', 'patient019',
    'patient020', 'patient021', 'patient022', 'patient023', 'patient024', 'patient025', 'patient026', 'patient027',
    'patient028', 'patient029', 'patient030', 'patient031', 'patient032', 'patient033', 'patient034', 'patient035',
    'patient036', 'patient037', 'patient038', 'patient039', 'patient040', 'patient041', 'patient042', 'patient044',
    'patient045', 'patient046', 'patient047', 'patient048', 'patient050', 'patient051', 'patient054', 'patient056',
    'patient060', 'patient065', 'patient066', 'patient067', 'patient069', 'patient071', 'patient072', 'patient075',
    'patient076', 'patient077', 'patient078', 'patient081', 'patient082', 'patient083', 'patient084',
    'patient085', 'patient086', 'patient087', 'patient088', 'patient089', 'patient090', 'patient091', 'patient092',
    'patient094', 'patient095', 'patient096', 'patient097', 'patient098', 'patient099', 'patient100', 'patient101',
    'patient102', 'patient152', 'patient158', 'patient163', 'patient223', 'patient230', 'patient231', 'patient268',
    'patient270', 'patient290' 
]

# ecg > 60 días del infarto (healed) SIN VT/VF
PTBMI60NOVTVF = [
    'patient005e', 'patient015c', 'patient022c', 'patient025c', 'patient027c', 'patient030d', 'patient032d',
    'patient033d', 'patient034d', 'patient035d', 'patient038c', 'patient039c', 'patient040d', 'patient041d',
    'patient042d', 'patient045d', 'patient048d', 'patient050d', 'patient051d', 'patient054d', 
    'patient065d', 'patient066c', 'patient067c', 'patient069d', 'patient072d', 'patient075d', 'patient076d',
    'patient078d', 'patient081d', 'patient082d', 'patient084d', 'patient085d', 'patient088d', 'patient090d',
    'patient091d', 'patient092d', 'patient094d', 'patient095d', 'patient099d'
]

# ecg < 7 días del infarto (healing) CON VT/VF
PTBMI07VTVF = [
    'patient043', 'patient049', 'patient053', 'patient073b', 'patient074b', 'patient080', 'patient197'
]

# ecg > 60 días del infarto (healed) CON VT/VF
PTBMI60VTVF = [
    'patient043c', 'patient049d', 'patient073d', 'patient074d', 'patient080d', 'patient111', 'patient197b'
]

# ecg < 7 días del infarto (healing) SIN VT/VF ANTERIOR
PTBMI07NOVTVFANT = [
    'patient004', 'patient005', 'patient006', 'patient007', 'patient010',
    'patient013', 'patient014', 
    'patient020', 'patient024', 'patient025', 'patient027',
    'patient028', 'patient029', 'patient030', 'patient032', 'patient033', 'patient034',
    'patient036', 'patient037', 'patient038', 'patient039', 'patient042', 'patient044',
    'patient046', 'patient047', 'patient048', 'patient051', 'patient056',
    'patient060', 'patient069',
    'patient076', 'patient081', 'patient082', 'patient083', 'patient084',
    'patient091', 
    'patient094', 'patient096', 'patient097', 'patient099', 'patient101',
    'patient158', 'patient163', 'patient223',
    'patient270', 'patient290'
]

# ecg > 60 días del infarto (healed) SIN VT/VF ANTERIOR
PTBMI60NOVTVFANT = [
    'patient005e', 'patient025c', 'patient027c', 'patient030d', 'patient032d',
    'patient033d', 'patient034d', 'patient038c', 'patient039c',
    'patient042d', 'patient048d', 'patient051d', 
    'patient069d', 'patient076d',
    'patient081d', 'patient082d', 'patient084d',
    'patient091d', 'patient094d', 'patient099d'
]

# ecg < 7 días del infarto (healing) SIN VT/VF INFERIOR
PTBMI07NOVTVFINF = [
    'patient001', 'patient008', 'patient011',
    'patient012', 'patient015', 'patient016','patient017', 'patient018', 'patient019',
    'patient021', 'patient022', 'patient023', 'patient026',
    'patient031', 'patient035',
    'patient040', 'patient041',
    'patient045', 'patient050', 'patient054',
    'patient065', 'patient066', 'patient067', 'patient071', 'patient072', 'patient075',
    'patient077', 'patient078',
    'patient086', 'patient087', 'patient088', 'patient089', 'patient090', 'patient092',
    'patient095', 'patient098', 'patient100',
    'patient102', 'patient152', 'patient230', 'patient231', 'patient268'
]

# ecg > 60 días del infarto (healed) SIN VT/VF INFERIOR
PTBMI60NOVTVFINF = [
    'patient015c', 'patient022c',
    'patient035d', 'patient040d', 'patient041d',
    'patient045d', 'patient050d', 'patient054d',
    'patient065d', 'patient066c', 'patient067c', 'patient072d', 'patient075d',
    'patient078d', 'patient088d', 'patient090d',
    'patient092d', 'patient095d'
]

# Unir todas las listas en una sola base de datos
ALLPTB = PTBHE + PTBMI07NOVTVF + PTBMI60NOVTVF + PTBMI07VTVF + PTBMI60VTVF + PTBMI07NOVTVFANT + PTBMI60NOVTVFANT + PTBMI07NOVTVFINF + PTBMI60NOVTVFINF

# Guardar en un archivo
with open('ptbaprocesar.pkl', 'wb') as f:
    pickle.dump({
        'ALLPTB': ALLPTB,
        'PTBHE': PTBHE,
        'PTBMI07NOVTVF': PTBMI07NOVTVF,
        'PTBMI60NOVTVF': PTBMI60NOVTVF,
        'PTBMI07VTVF': PTBMI07VTVF,
        'PTBMI60VTVF': PTBMI60VTVF,
        'PTBMI07NOVTVFANT': PTBMI07NOVTVFANT,
        'PTBMI60NOVTVFANT': PTBMI60NOVTVFANT,
        'PTBMI07NOVTVFINF': PTBMI07NOVTVFINF,
        'PTBMI60NOVTVFINF': PTBMI60NOVTVFINF
    }, f)

#-------------------------------------------------

# Cargar base de datos PTB XL
data = scipy.io.loadmat('ptbaprocesarPTBXL.mat')

# Extraer la variable
PTBHE_XL = data['PTBHE'][0]
PTBMI07NOVTVF_XL = data['PTBMI07NOVTVF'][0]
PTBMI60NOVTVF_XL = data['PTBMI60NOVTVF'][0]
PTBMI07VTVF_XL = data['PTBMI07VTVF'][0]
PTBMI60VTVF_XL = data['PTBMI60VTVF'][0]
PTBMI07NOVTVFANT_XL = data['PTBMI07NOVTVFANT'][0]
PTBMI60NOVTVFANT_XL = data['PTBMI60NOVTVFANT'][0]
PTBMI07NOVTVFINF_XL = data['PTBMI07NOVTVFINF'][0]
PTBMI60NOVTVFINF_XL = data['PTBMI60NOVTVFINF'][0]

# Convertir arrays a listas
PTBHE_XL = [paciente[0] for paciente in PTBHE_XL]
PTBMI07NOVTVF_XL = [paciente[0] for paciente in PTBMI07NOVTVF_XL]
PTBMI60NOVTVF_XL = [paciente[0] for paciente in PTBMI60NOVTVF_XL]
PTBMI07VTVF_XL = [paciente[0] for paciente in PTBMI07VTVF_XL]
PTBMI60VTVF_XL = [paciente[0] for paciente in PTBMI60VTVF_XL]
PTBMI07NOVTVFANT_XL = [paciente[0] for paciente in PTBMI07NOVTVFANT_XL]
PTBMI60NOVTVFANT_XL = [paciente[0] for paciente in PTBMI60NOVTVFANT_XL]
PTBMI07NOVTVFINF_XL = [paciente[0] for paciente in PTBMI07NOVTVFINF_XL]
PTBMI60NOVTVFINF_XL = [paciente[0] for paciente in PTBMI60NOVTVFINF_XL]

# Unir los grupos de ambas bases de datos
PTBHE = PTBHE + PTBHE_XL
PTBMI07NOVTVF = PTBMI07NOVTVF + PTBMI07NOVTVF_XL
PTBMI60NOVTVF = PTBMI60NOVTVF + PTBMI60NOVTVF_XL
PTBMI07VTVF = PTBMI07VTVF + PTBMI07VTVF_XL
PTBMI60VTVF = PTBMI60VTVF + PTBMI60VTVF_XL
PTBMI07NOVTVFANT = PTBMI07NOVTVFANT + PTBMI07NOVTVFANT_XL
PTBMI60NOVTVFANT = PTBMI60NOVTVFANT + PTBMI60NOVTVFANT_XL
PTBMI07NOVTVFINF = PTBMI07NOVTVFINF + PTBMI07NOVTVFINF_XL
PTBMI60NOVTVFINF = PTBMI60NOVTVFINF + PTBMI60NOVTVFINF_XL

# Unir todas las listas en una sola base de datos
ALLPTB = PTBHE + PTBMI07NOVTVF + PTBMI60NOVTVF + PTBMI07VTVF + PTBMI60VTVF + PTBMI07NOVTVFANT + PTBMI60NOVTVFANT + PTBMI07NOVTVFINF + PTBMI60NOVTVFINF

# Guardar en un archivo
with open('ptbaprocesar.pkl', 'wb') as f:
    pickle.dump({
        'ALLPTB': ALLPTB,
        'PTBHE': PTBHE,
        'PTBMI07NOVTVF': PTBMI07NOVTVF,
        'PTBMI60NOVTVF': PTBMI60NOVTVF,
        'PTBMI07VTVF': PTBMI07VTVF,
        'PTBMI60VTVF': PTBMI60VTVF,
        'PTBMI07NOVTVFANT': PTBMI07NOVTVFANT,
        'PTBMI60NOVTVFANT': PTBMI60NOVTVFANT,
        'PTBMI07NOVTVFINF': PTBMI07NOVTVFINF,
        'PTBMI60NOVTVFINF': PTBMI60NOVTVFINF
    }, f)
    
    datos={
   'ALLPTB': ALLPTB,
   'PTBHE': PTBHE,
   'PTBMI07NOVTVF': PTBMI07NOVTVF,
   'PTBMI60NOVTVF': PTBMI60NOVTVF,
   'PTBMI07VTVF': PTBMI07VTVF,
   'PTBMI60VTVF': PTBMI60VTVF,
   'PTBMI07NOVTVFANT': PTBMI07NOVTVFANT,
   'PTBMI60NOVTVFANT': PTBMI60NOVTVFANT,
   'PTBMI07NOVTVFINF': PTBMI07NOVTVFINF,
   'PTBMI60NOVTVFINF': PTBMI60NOVTVFINF
}

# Guardar en un archivo .mat
savemat('ptbaprocesar11.mat', datos)

DI = 1
