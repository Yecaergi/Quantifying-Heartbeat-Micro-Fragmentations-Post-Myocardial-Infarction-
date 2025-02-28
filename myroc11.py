import numpy as np
from scipy import interpolate
from sklearn.metrics import roc_curve

class MyROC:
    @staticmethod
    def ROC(grSanos, grEnfer):
        stats = {
            'FPR': None,       # 1-especificidad: FP/(TN+FP) false positive rate, fallout
            'TPR': None,       # sensibilidad:  TP/(TP+FN) true positive rate, sensitivity, recall
            'T': None,         # array of thresholds on classifier scores for the computed values
            'OPTROCPT': None,  # the optimal operating point of the ROC curve
            'INDEXOPT': None,  # index of the optimal operating point of the ROC curve
            'SEN': None,       # sensibilidad del punto óptimo TP/(TP+FN)
            'ESP': None,       # 1-especificidad del punto óptimo 1-TN/(TN+FP)
            'AUC': None,       # area under curve (AUC) for the computed values of X and Y
            'TH': None         # umbral en el punto optimo
        }
        
        # argumentos de entrada
        # grSanos: [a] vector columna con los datos de los sujetos sanos
        # grEnfer: [b] vector columna con los datos de los sujetos enfermos
 
        # convierto de una matriz de [n x m] a un vector columna de [(n*m) x 1]
        # (concatena todas las columnas)
        
        grSanos = np.array(grSanos).flatten()
        grEnfer = np.array(grEnfer).flatten()
        if np.mean(grSanos) >= np.mean(grEnfer):
            grSanos, grEnfer = grEnfer, grSanos

        scores = np.concatenate([grSanos, grEnfer])
        labels = np.concatenate([np.zeros_like(grSanos), np.ones_like(grEnfer)])

        mask = ~np.isnan(scores)
        scores = scores[mask]
        labels = labels[mask]

        # obtengo sensibilidad, especificidad
        FPR, TPR, T = roc_curve(labels, scores)

        st = MyROC.puntoOptimo(FPR, TPR)
        TH = T[st['INDEXOPT']]

        # asigno los resultados a la estructura de salida
        stats.update({
            'FPR': FPR, 
            'TPR': TPR, 
            'T': T, 
            'OPTROCPT': st['OPTROCPT'],
            'INDEXOPT': st['INDEXOPT'],
            'SEN': st['SEN'], 
            'ESP': st['ESP'],
            'AUC': st['AUC'], 
            'TH': TH
        })

        return stats

    @staticmethod
    def jROC(grSanos, grEnfer):
        stats = {
            'FPR': None,       # 1-especificidad: FP/(TN+FP) false positive rate
            'TPR': None,       # sensibilidad:  TP/(TP+FN) true positive rate
            'OPTROCPT': None,  # the optimal operating point of the ROC curve
            'INDEXOPT': None,  # index of the optimal operating point of the ROC curve
            'SEN': None,       # TP/(TP+FN) sensibilidad del punto óptimo
            'ESP': None,       # TN/(TN+FP) especificidad del punto óptimo
            'AUC': None,       # area under curve (AUC) for the computed values of X and Y
            'TH': None         # umbral en el punto optimo
        }
        
        # argumentos de entrada
        # grSanos: [a1 a2] matriz doble columna con los pares de datos de los sujetos sanos
        # grEnfer: [b1 b2] matriz doble columna con los pares de datos de los sujetos enfermos

        sta = MyROC.ROC(grSanos[:, 0], grEnfer[:, 0])
        stb = MyROC.ROC(grSanos[:, 1], grEnfer[:, 1])

        # calculo las combinaciones A, B, A*B y A+B, interpolo para que sean de la misma longitud
        common_fpr = np.sort(np.unique(np.concatenate([sta['FPR'], stb['FPR']])))
        tpr_a = np.interp(common_fpr, sta['FPR'], sta['TPR'])
        tpr_b = np.interp(common_fpr, stb['FPR'], stb['TPR'])

        fp1, tp1 = common_fpr * common_fpr, tpr_a * tpr_b
        fp2, tp2 = common_fpr, tpr_a
        fp3, tp3 = common_fpr, tpr_b
        fp4, tp4 = fp2 + fp3 - fp2 * fp3, tp2 + tp3 - tp2 * tp3

        # elimino las repeticiones en fp... y la dejo monotonicamente creciente
        fp1, tp1 = MyROC.monotonic(fp1, tp1)
        fp2, tp2 = MyROC.monotonic(fp2, tp2)
        fp3, tp3 = MyROC.monotonic(fp3, tp3)
        fp4, tp4 = MyROC.monotonic(fp4, tp4)

        # obtengo una unica fp a partir de las 4 fp's (merge - concatenar)
        fp = np.sort(np.unique(np.concatenate([fp1, fp2, fp3, fp4])))

        # interpolo todos los pares fpx-tpx de manera de coincidan con fp
        tx1 = np.interp(fp, fp1, tp1)
        tx2 = np.interp(fp, fp2, tp2)
        tx3 = np.interp(fp, fp3, tp3)
        tx4 = np.interp(fp, fp4, tp4)
        tx = np.column_stack([tx1, tx2, tx3, tx4])

        # divido en bins y busco el par (fp,tp) optimo en cada bin. Luego
        # interpolo una curva que pase por todos esos pares y busco el
        # punto optimo de toda la interpolacion
        n = len(fp)
        bins = n // 20
        tt = np.linspace(0, n-1, bins, dtype=int)
        fi = []
        ti = []
        for i in range(bins - 1):
            t = slice(tt[i], tt[i+1])
            st1 = MyROC.puntoOptimo(fp[t], tx1[t])
            st2 = MyROC.puntoOptimo(fp[t], tx2[t])
            st3 = MyROC.puntoOptimo(fp[t], tx3[t])
            st4 = MyROC.puntoOptimo(fp[t], tx4[t])
            opt_tpr = max(st1['OPTROCPT'][1], st2['OPTROCPT'][1], st3['OPTROCPT'][1], st4['OPTROCPT'][1])
            ti.append(opt_tpr)
            opt_fpr = [st['OPTROCPT'][0] for st in [st1, st2, st3, st4]][np.argmax([st['OPTROCPT'][1] for st in [st1, st2, st3, st4]])]
            fi.append(opt_fpr)
            
        # agrego comienzo y fin para poder interpolar desde 0 hasta 1
        fi = np.array([fp[0]] + fi + [fp[-1]])
        ti = np.array([min(tx[0])] + ti + [max(tx[-1])])
        fi, ti = MyROC.monotonic(fi, ti)
        tp = np.interp(fp, fi, ti)

        # obtengo sensibilidad, especificidad y AUC
        st = MyROC.puntoOptimo(fp, tp)
        st['INDEXOPT'] = int(st['INDEXOPT'] * len(sta['FPR']) / len(fp))
        TH1 = sta['T'][st['INDEXOPT']]
        TH2 = stb['T'][st['INDEXOPT']]

        # asigno los resultados a la estructura de salida
        stats.update({
            'FPR': fp, 
            'TPR': tp, 
            'OPTROCPT': st['OPTROCPT'],
            'INDEXOPT': st['INDEXOPT'], 
            'SEN': st['SEN'], 
            'ESP': st['ESP'],
            'AUC': st['AUC'], 
            'TH': [TH1, TH2]
        })

        return stats

    @staticmethod
    def puntoOptimo(fp, tp):
        pos = np.column_stack([fp, tp])
        zo = np.column_stack([np.linspace(0, 1, len(fp)), np.linspace(0, 1, len(fp))])
        dist = np.sqrt(np.sum((pos - zo)**2, axis=1))
        INDEXOPT = np.argmax(dist)      # maxima distancia entre la curva y la diagonal [0,1] 
        OPTROCPT = pos[INDEXOPT]

        # el AUC se calcula por el coeficiente de Gini (asi hace matlab
        # en la funcion perfcurve) 2 * AUC = G + 1, 
        # donde G = abs(1 - sum_k((fp(k)-fp(k-1))*(tp(k)+tp(k-1))))
        G = np.abs(1 - np.sum((fp[1:] - fp[:-1]) * (tp[1:] + tp[:-1])))
        AUC = 0.5 * (G + 1)

        return {
            'dist': dist,
            'INDEXOPT': INDEXOPT,
            'OPTROCPT': OPTROCPT,
            'SEN': 100 * OPTROCPT[1],
            'ESP': 100 * (1 - OPTROCPT[0]),
            'AUC': AUC
        }

    @staticmethod
    def monotonic(fx, tx=None):
        fx, idx = np.unique(fx, return_index=True)
        sort_idx = np.argsort(fx)
        fx = fx[sort_idx]
        if tx is not None:
            tx = tx[idx][sort_idx]
            return fx, tx
        return fx

