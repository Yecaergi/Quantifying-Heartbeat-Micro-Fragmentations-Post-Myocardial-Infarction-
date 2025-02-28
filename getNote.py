def getNote(notes, origen):
    # Valores considerados como nan
    nan_values = {'n/a', 'No stenoses', 'No stenosis', 'not available', 'no', 'unknown'}
    
    # Lista para almacenar las notas resultantes
    vnota = [None] * len(notes)
    
    for i, note in enumerate(notes):
        # Buscar la coincidencia del origen
        index = [origen in n for n in note[0]]
        
        if any(index):
            ind = index.index(True)
            nota = note[1][ind]
            
            # Verificar si la nota es un valor para nan
            if nota in nan_values:
                vnota[i] = float('nan')
            else:
                vnota[i] = nota
        else:
            # Si no se encuentra el origen, asignar nan
            vnota[i] = float('nan')
    
    return vnota



