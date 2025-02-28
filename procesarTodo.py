from procesarTodoH import procesarTodoH
from ptbaprocesar import PTBHE, PTBMI07NOVTVF, PTBMI60NOVTVF, PTBMI07VTVF, PTBMI60VTVF, PTBMI07NOVTVFANT, PTBMI60NOVTVFANT, PTBMI07NOVTVFINF, PTBMI60NOVTVFINF, ALLPTB


Fs=1000;
inicioVentana = [0.08, 0.35, -0.1];
ventanaSize   = [0.256, 0.128, 0.3];
subfolder = ['QRS', 'lineaBase', 'ondaT']

# poner 1 (QRS) o 2 (linea de base)
procesarTodoH(1, subfolder=subfolder[0], wName='db6', wMaxScale=16, wTscale='continua', 
                  procesar='tiempo', recortar=False)
    

print('\n\n')




