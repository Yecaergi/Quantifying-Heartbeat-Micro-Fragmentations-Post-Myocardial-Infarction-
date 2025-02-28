import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

def miAnova(g1, g2, grNames, varNames):
    # Concatenar los grupos y crear nombres de grupos
    ng1 = g1.shape[0]
    ng2 = g2.shape[0]
    grupos = np.vstack([g1, g2])
    gruposName = np.array([grNames[0]] * ng1 + [grNames[1]] * ng2)
    
    # Crear un DataFrame
    df = pd.DataFrame(grupos, columns=varNames)
    df['grupos'] = gruposName
    
    # Realizar ANOVA univariado para cada variable dependiente
    results = {}
    for var in varNames:
        model = ols(f'{var} ~ grupos', data=df).fit()
        anova_results = anova_lm(model)
        results[var] = anova_results
    
    return results


