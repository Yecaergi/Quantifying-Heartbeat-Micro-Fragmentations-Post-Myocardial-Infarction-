import numpy as np
import matplotlib.pyplot as plt

def CotasComplejidad(M, *args):
    # Funciones adicionales para Jensen-Shannon
    def h1(x, n1, m):
        return -1/np.log(n1+np.finfo(float).eps) * (x*np.log(x+np.finfo(float).eps) + (1-x)*np.log((1-x+np.finfo(float).eps)/(n1-m-1)))

    def desej(x, n1, m, alfa=0.5):
        aux0 = (1 - alfa) / n1
        aux1 = alfa * x + aux0
        aux2 = alfa * (1-x)/(n1-m-1) + aux0
        return -(aux1 * np.log(aux1+np.finfo(float).eps) + (n1-m-1) * aux2 * np.log(aux2+np.finfo(float).eps) \
                 + m * aux0 * np.log(aux0+np.finfo(float).eps))

    def h1r(x, n1, m, q):
        return 1/np.log(n1+np.finfo(float).eps) / (1-q) * \
               (np.log(x**q+np.finfo(float).eps) + np.log(((1-x**q)+np.finfo(float).eps)/(n1-m-1)))

    def KR(pm, qm, q):
        return np.log(pm * (pm/qm)**(q-1)+np.finfo(float).eps)/(q-1)

    # Definición de variables iniciales
    cualentropia = 'SHANNON'
    isplot = True
    q = 2

    # Procesar argumentos variables
    nVarargs = len(args)
    if nVarargs > 0:
        k = 0
        while k < nVarargs:
            if args[k] == 'cualentropia':
                cualentropia = args[k+1]
                args = args[:k] + args[k+2:]
                nVarargs -= 2
            elif args[k] == 'Q':
                q = args[k+1]
                args = args[:k] + args[k+2:]
                nVarargs -= 2
            elif args[k] == 'isplot':
                isplot = args[k+1]
                args = args[:k] + args[k+2:]
                nVarargs -= 2
            elif args[k] == 'pe' or args[k] == 'Sim':
                args = args[:k] + args[k+2:]
                nVarargs -= 2
            else:
                k += 1

    # Eliminar argumentos restantes que no van al plot
    args = [arg for arg in args if arg not in ['cualentropia', 'Q', 'isplot', 'pe', 'Sim']]
    nVarargs = len(args)

    # Calculo de las cotas
    if cualentropia.upper() == 'SHANNON':
        N = 256
        Qo = -2 / ((M+1)/M * np.log(M+1+np.finfo(float).eps) - 2*np.log(2*M+np.finfo(float).eps) + np.log(M+np.finfo(float).eps))

        # Cota inferior continua
        X1 = np.linspace(1/M, 1, N)
        HHmax = h1(X1, M, 0)
        Dmax = desej(X1, M, 0) - 0.5 * np.log(M+np.finfo(float).eps) * h1(X1, M, 0) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmax = Qo * HHmax * Dmax
        HHmax = HHmax[~np.isnan(HHmax)]  # Elimino los puntos donde queda 0*log(0)
        CCmax = CCmax[~np.isnan(CCmax)]  # Elimino los puntos donde queda 0*log(0)

        # Cota superior a trazos y envolvente
        X1 = np.array([])
        in_vals = np.array([], dtype=int)
        for i in range(M-2, -1, -1):
            Wi = 1/(M-i)
            kk = int(np.ceil(Wi*N))
            in_vals = np.concatenate((in_vals, np.array([i]*kk)))
            X1 = np.concatenate((X1, np.linspace(0, Wi, kk)))

        HHmin = h1(X1, M, in_vals)
        Dmin = desej(X1, M, in_vals) - 0.5 * np.log(M+np.finfo(float).eps) * h1(X1, M, in_vals) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmin = Qo * Dmin * HHmin
        HHmin = HHmin[~np.isnan(HHmin)]  # Elimino los puntos donde queda 0*log(0)
        CCmin = CCmin[~np.isnan(CCmin)]  # Elimino los puntos donde queda 0*log(0)

        # Envolvente de la cota a trazos mediante ajuste polinómico
        HHmine = h1(np.array([0]), M, np.array([M-2]))
        Dmine = desej(np.array([0]), M, np.array([M-2])) - 0.5 * np.log(M+np.finfo(float).eps) * h1(np.array([0]), M, np.array([M-2])) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmine = Qo * Dmine * HHmine
        HHmine = np.concatenate(([HHmin[0]], HHmine))  # Agrego el inicio (0,0) a los trazos
        CCmine = np.concatenate(([CCmin[0]], CCmine))  # Agrego el inicio (0,0) a los trazos

        if M > 3:
            ordenP = 3
        else:
            ordenP = 2
     
        P = np.polyfit(HHmine, CCmine, ordenP)
        HHmine = np.linspace(HHmine[0], HHmine[-1], N)
        CCmine = np.polyval(P, HHmine)
        CCmine[CCmine < 0] = 0

    elif cualentropia.upper() == 'EUCLIDEO':
        N = 256
        Xe = 1/M
        Qo = M/(M-1)

        # Cota máxima
        X1 = np.linspace(1/M, 1, N)
        HHmax = -(1/np.log(M+np.finfo(float).eps)) * (X1*np.log(X1+np.finfo(float).eps) + (1-X1)*np.log((1-X1)/(M-1)+np.finfo(float).eps))
        Dmax = (X1-Xe)**2 + (M-1)*(((1-X1)/(M-1))-Xe)**2
        CCmax = Qo * Dmax * HHmax
        HHmax = HHmax[~np.isnan(HHmax)]  # Elimino los puntos donde queda 0*log(0)
        CCmax = CCmax[~np.isnan(CCmax)]  # Elimino los puntos donde queda 0*log(0)

        # Cota mínima a trazos y envolvente
        X1 = np.array([])
        in_vals = np.array([], dtype=int)
        for i in range(M-2, -1, -1):
            Wi = 1/(M-i)
            kk = int(np.ceil(Wi*N))
            in_vals = np.concatenate((in_vals, np.array([i]*kk)))
            X1 = np.concatenate((X1, np.linspace(0, Wi, kk)))

        HHmin = -(1/np.log(M+np.finfo(float).eps)) * (X1*np.log(X1+np.finfo(float).eps) + (1-X1)*np.log((1-X1)/(M-in_vals-1)+np.finfo(float).eps))
        Dmin = (X1-Xe)**2 + (M-in_vals-1)*(((1-X1)/(M-in_vals-1))-Xe)**2 + in_vals*Xe**2
        CCmin = Qo * Dmin * HHmin
        HHmin = HHmin[~np.isnan(HHmin)]  # Elimino los puntos donde queda 0*log(0)
        CCmin = CCmin[~np.isnan(CCmin)]  # Elimino los puntos donde queda 0*log(0)

        # Envolvente de la cota a trazos
        HHmine = HHmin
        Dmine = np.exp(-HHmine*np.log(M)) - 1/M
        CCmine = Qo * Dmine * HHmine

    elif cualentropia.upper() == 'RENYI':
        N = 256
        Xe = 1/M

        Qo = 2*(q-1)*(np.log(((M+1)**(1-q)+(M-1))/(2**(1-q)*M)+np.finfo(float).eps) + \
            (1-q)*np.log((M+1)/(2*M)+np.finfo(float).eps))**(-1)

        # Cota inferior continua
        X1 = np.linspace(1/M, 1, N)
        HHmax = h1(X1, M, 0)
        Dmax = desej(X1, M, 0) - 0.5 * h1(X1, M, 0) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmax = Qo * HHmax * Dmax
        HHmax = HHmax[~np.isnan(HHmax)]  # Elimino los puntos donde queda 0*log(0)
        CCmax = CCmax[~np.isnan(CCmax)]  # Elimino los puntos donde queda 0*log(0)

        # Cota superior a trazos y envolvente
        X1 = np.array([])
        in_vals = np.array([], dtype=int)
        for i in range(M-2, -1, -1):
            Wi = 1/(M-i)
            kk = int(np.ceil(Wi*N))
            in_vals = np.concatenate((in_vals, np.array([i]*kk)))
            X1 = np.concatenate((X1, np.linspace(0, Wi, kk)))

        HHmin = h1(X1, M, in_vals)
        Dmin = desej(X1, M, in_vals) - 0.5 * h1(X1, M, in_vals) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmin = Qo * Dmin * HHmin
        HHmin = HHmin[~np.isnan(HHmin)]  # Elimino los puntos donde queda 0*log(0)
        CCmin = CCmin[~np.isnan(CCmin)]  # Elimino los puntos donde queda 0*log(0)

        # Envolvente de la cota a trazos mediante ajuste polinómico
        HHmine = h1(X1, M, in_vals)
        Dmine = desej(X1, M, in_vals) - 0.5 * h1(X1, M, in_vals) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmine = Qo * Dmine * HHmine
        HHmine = np.concatenate(([HHmin[0]], HHmine))  # Agrego el inicio (0,0) a los trazos
        CCmine = np.concatenate(([CCmin[0]], CCmine))  # Agrego el inicio (0,0) a los trazos

     
        P = np.polyfit(HHmine, CCmine, 3)
        HHmine = np.linspace(HHmine[0], HHmine[-1], N)
        CCmine = np.polyval(P, HHmine)
        CCmine[CCmine < 0] = 0

    elif cualentropia.upper() == 'TSALLIS':
        N = 256
        Qo = (1-q)*(1-((1+M**q)*(1+M)**(1-q)+(M-1))/(2**(2-q)*M))**(-1)

        # Cota inferior continua
        X1 = np.linspace(1/M, 1, N)
        HHmax = h1(X1, M, 0)
        Dmax = desej(X1, M, 0) - 0.5 * h1(X1, M, 0) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmax = Qo * HHmax * Dmax
        HHmax = HHmax[~np.isnan(HHmax)]  # Elimino los puntos donde queda 0*log(0)
        CCmax = CCmax[~np.isnan(CCmax)]  # Elimino los puntos donde queda 0*log(0)

        # Cota superior a trazos y envolvente
        X1 = np.array([])
        in_vals = np.array([], dtype=int)
        for i in range(M-2, -1, -1):
            Wi = 1/(M-i)
            kk = int(np.ceil(Wi*N))
            in_vals = np.concatenate((in_vals, np.array([i]*kk)))
            X1 = np.concatenate((X1, np.linspace(0, Wi, kk)))

        HHmin = h1(X1, M, in_vals)
        Dmin = desej(X1, M, in_vals) - 0.5 * h1(X1, M, in_vals) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmin = Qo * Dmin * HHmin
        HHmin = HHmin[~np.isnan(HHmin)]  # Elimino los puntos donde queda 0*log(0)
        CCmin = CCmin[~np.isnan(CCmin)]  # Elimino los puntos donde queda 0*log(0)

        # Envolvente de la cota a trazos mediante ajuste polinómico
        HHmine = h1(X1, M, in_vals)
        Dmine = desej(X1, M, in_vals) - 0.5 * h1(X1, M, in_vals) * np.log(M+np.finfo(float).eps) - 0.5 * np.log(M+np.finfo(float).eps)
        CCmine = Qo * Dmine * HHmine
        HHmine = np.concatenate(([HHmin[0]], HHmine))  # Agrego el inicio (0,0) a los trazos
        CCmine = np.concatenate(([CCmin[0]], CCmine))  # Agrego el inicio (0,0) a los trazos
  
        P = np.polyfit(HHmine, CCmine, 3)
        HHmine = np.linspace(HHmine[0], HHmine[-1], N)
        CCmine = np.polyval(P, HHmine)
        CCmine[CCmine < 0] = 0

    # Graficar las cotas
    if isplot:
        plt.plot(HHmax, CCmax, *args)
        plt.plot(HHmin, CCmin, *args)
        # plt.plot(HHmine, CCmine, *args)  # Opcional: plotear la envolvente de la cota a trazos
        plt.xlabel('Entropia')
        plt.ylabel('Complejidad')
        #plt.title('Cotas de Complejidad')
        plt.grid(True)
        plt.show()

    return HHmax, CCmax, HHmin, CCmin


# Ejemplo de uso
#HHmax, CCmax, HHmin, CCmin = CotasComplejidad(40, 'cualentropia', 'SHANNON', 'isplot', True)
