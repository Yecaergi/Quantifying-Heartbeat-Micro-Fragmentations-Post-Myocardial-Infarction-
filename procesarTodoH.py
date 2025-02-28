import os
import numpy as np
from entropiaWavelet import entropiaWavelet
from ptbaprocesar import PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF 
import scipy.io as sio



def clean_none(data):
    """
    Replaces None in lists or dicts with an appropriate value (e.g., np.nan or empty string).
    """
    if isinstance(data, dict):
        return {k: clean_none(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_none(x) for x in data]
    elif data is None:
        return np.nan  # or '' for strings
    else:
        return data

def procesarTodoH(cc, *args, **kwargs):
    subfolder = '\\'
    if 'subfolder' in kwargs:
        subfolder = kwargs['subfolder']
    
    if cc > 3 or cc < 1:
        print(f'No existe el grupo {cc} en este proceso.')
        return
    
    if cc == 1:
        outfileName = 'ctrl-healing-healed sin VTVF QRS Entropia'
        G1_NAME = 'CTRL'
        G2_NAME = 'MI7 sin VT/VF QRS'
        G3_NAME = 'MI60 sin VT/VF QRS'
        cualgrupo = [PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF]
    elif cc == 2:
        outfileName = 'ctrl-healing-healed sin VTVF lineaBase Entropia'
        G1_NAME = 'CTRL'
        G2_NAME = 'MI7 sin VT/VF Base'
        G3_NAME = 'MI60 sin VT/VF Base'
        cualgrupo = [PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF]
    elif cc == 3:
        outfileName = 'ctrl-healing-healed sin VTVF ondaT Entropia'
        G1_NAME = 'CTRL'
        G2_NAME = 'MI7 sin VT/VF ondaT'
        G3_NAME = 'MI60 sin VT/VF ondaT'
        cualgrupo = [PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF]

    cualgrupoName = [G1_NAME, G2_NAME, G3_NAME]
    nleads = 13  # es 15 si van las ortogonales
    leadNames = [None] * nleads

    nanH = {
        'mWd': np.nan, 'mDd': np.nan, 'mHd': np.nan, 'sHd': np.nan, 'mCd': np.nan,
        'sCd': np.nan, 'mEt': np.nan, 'rmsnoise': np.nan, 'diagnose': np.nan,
        'filas': np.nan, 'units': np.nan, 'fileName': np.nan, 'wName': np.nan,
        'wScales': np.nan, 'wNscales': np.nan, 'wMaxScale': np.nan,
        'wTscale': np.nan, 'procesar': np.nan
    }

    # Inicializar estructuras
    nGrupo1 = len(cualgrupo[0])
    Hgrupo1 = np.full((nGrupo1, nleads), nanH)
    rmsnoisegrupo1 = np.full((len(cualgrupo[0]), nleads), np.nan)
    grupo1Name = [None] * len(cualgrupo[0])
    grupo1OriginalName = [None] * len(cualgrupo[0])
    notes1 = [None] * len(cualgrupo[0])
    indexG1 = 0

    nGrupo2 = len(cualgrupo[1])
    Hgrupo2 = np.full((nGrupo2, nleads), nanH)
    rmsnoisegrupo2 = np.full((len(cualgrupo[1]), nleads), np.nan)
    grupo2Name = [None] * len(cualgrupo[1])
    grupo2OriginalName = [None] * len(cualgrupo[1])
    notes2 = [None] * len(cualgrupo[1])
    indexG2 = 0

    nGrupo3 = len(cualgrupo[2])
    Hgrupo3 = np.full((nGrupo3, nleads), nanH)
    rmsnoisegrupo3 = np.full((len(cualgrupo[2]), nleads), np.nan)
    grupo3Name = [None] * len(cualgrupo[2])
    grupo3OriginalName = [None] * len(cualgrupo[2])
    notes3 = [None] * len(cualgrupo[2])
    indexG3 = 0

    print(f'\nprocesando {outfileName}...\n')

    ngrupos = len(cualgrupo)
    for grupo in range(ngrupos):
        dirs = cualgrupo[grupo]
        ndirs = len(dirs)

        print(f'\nprocesando el grupo {cualgrupoName[grupo]}...\n')
        for i, thisdir in enumerate(dirs):
            carpeta = os.path.join(os.getcwd(), 'Resultados', subfolder, thisdir)
            if not os.path.isdir(carpeta):
                print(f'No se encuentra el paciente: {thisdir}  VERIFIQUE !!!!')
                continue

            files = [f for f in os.listdir(carpeta) if f.endswith('.mat')]
            nfiles = len(files)
            print(f'pac {i+1} de {ndirs} ({thisdir}): ', end='')

            if nfiles == 0:
                print('\n')
                continue

            
            for j, fname in enumerate(files):
                archivo = os.path.join(carpeta, fname)
                fid = sio.loadmat(archivo)
                lead = fid['lead'][0][0]
                
               # if lead >= nleads:
                #    print(f'Lead {lead} out of range. Skipping.')
                #    continue
                
                leadNames[lead] = fid['strderivacion'][0]
                print(f'{leadNames[lead]}, ', end='')

                H = entropiaWavelet(fid, *args)
                noise = H['rmsnoise']

                if cualgrupoName[grupo] == G1_NAME:
                    Hgrupo1[indexG1, lead] = H
                    rmsnoisegrupo1[indexG1, lead] = noise
                elif cualgrupoName[grupo] == G2_NAME:
                    Hgrupo2[indexG2, lead] = H
                    rmsnoisegrupo2[indexG2, lead] = noise
                elif cualgrupoName[grupo] == G3_NAME:
                    Hgrupo3[indexG3, lead] = H
                    rmsnoisegrupo3[indexG3, lead] = noise

            print('\n')

            notes = fid['notes']

            if cualgrupoName[grupo] == G1_NAME:
                grupo1Name[indexG1] = thisdir
                grupo1OriginalName[indexG1] = fid['originalFileName'][0]
                notes1[indexG1] = notes
                indexG1 += 1
            elif cualgrupoName[grupo] == G2_NAME:
                grupo2Name[indexG2] = thisdir
                grupo2OriginalName[indexG2] = fid['originalFileName'][0]
                notes2[indexG2] = notes
                indexG2 += 1
            elif cualgrupoName[grupo] == G3_NAME:
                grupo3Name[indexG3] = thisdir
                grupo3OriginalName[indexG3] = fid['originalFileName'][0]
                notes3[indexG3] = notes
                indexG3 += 1

    Hgrupo1 = Hgrupo1[:indexG1, :]
    rmsnoisegrupo1 = rmsnoisegrupo1[:indexG1, :]
    grupo1Name = grupo1Name[:indexG1]
    grupo1OriginalName = grupo1OriginalName[:indexG1]
    notes1 = notes1[:indexG1]
    nGrupo1 = len(Hgrupo1)

    Hgrupo2 = Hgrupo2[:indexG2, :]
    rmsnoisegrupo2 = rmsnoisegrupo2[:indexG2, :]
    grupo2Name = grupo2Name[:indexG2]
    grupo2OriginalName = grupo2OriginalName[:indexG2]
    notes2 = notes2[:indexG2]
    nGrupo2 = len(Hgrupo2)

    Hgrupo3 = Hgrupo3[:indexG3, :]
    rmsnoisegrupo3 = rmsnoisegrupo3[:indexG3, :]
    grupo3Name = grupo3Name[:indexG3]
    grupo3OriginalName = grupo3OriginalName[:indexG3]
    notes3 = notes3[:indexG3]
    nGrupo3 = len(Hgrupo3)

    print(f'\n{outfileName} finalizado.\n')

    # Guardar los resultados
    outfile = os.path.join(os.getcwd(), 'Resultados', subfolder, f'{outfileName}.mat')
    sio.savemat(outfile, clean_none({
        'Hgrupo1': Hgrupo1, 'rmsnoisegrupo1': rmsnoisegrupo1,
        'grupo1Name': grupo1Name, 'grupo1OriginalName': grupo1OriginalName,
        'notes1': notes1, 'nGrupo1': nGrupo1, 'grupo2Name': grupo2Name,
        'grupo2OriginalName': grupo2OriginalName, 'Hgrupo2': Hgrupo2,
        'rmsnoisegrupo2': rmsnoisegrupo2, 'notes2': notes2, 'nGrupo2': nGrupo2,
        'grupo3Name': grupo3Name, 'grupo3OriginalName': grupo3OriginalName,
        'Hgrupo3': Hgrupo3, 'rmsnoisegrupo3': rmsnoisegrupo3, 'notes3': notes3
    }))

