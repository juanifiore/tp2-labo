
#def mapa_de_calor(imagenes,funcion_columnas,n_columnas,k,k_folds, nombre_funcion):
#    k_vecinos = np.arange(1,k)
#    
#    resultados_test = np.zeros((k_folds,len(n_columnas), len(k_vecinos)))
#    
#    j = -1  # lo voy a usar para asignar en la posicion j de la matriz
#    for n in n_columnas:
#        j += 1
#        columnas_selec = funcion_columnas(imagenes,n)
#        PIXELES = columnas_selec.iloc[:,1:]
#        DIGITO = columnas_selec[0]
#        
#        kf = StratifiedKFold(n_splits=k_folds,shuffle=True) 
#        i=0
#        for train, test in kf.split(PIXELES,DIGITO):
#            X_train =   PIXELES.iloc[train]
#            Y_train = DIGITO.iloc[train]
#            X_test = PIXELES.iloc[test]
#            Y_test = DIGITO.iloc[test]
#            for k in k_vecinos:
#                model = KNeighborsClassifier(n_neighbors = k)
#                model.fit(X_train, Y_train)
#                predicciones_test = model.predict(X_test)
#                precision_test = metrics.accuracy_score(Y_test, predicciones_test)
#                resultados_test[i,j,k-1] = precision_test
#            i=i+1 # para indexar en cada foldeo y asignar en resultados_test
#
#    promedio_precisiones = np.mean(resultados_test,axis=0)
#
#    mayor_precision = np.max(promedio_precisiones)
#    print('\nMayor precision alcanzada: ',mayor_precision)
#    print()
#    columnas_mayor_precision,vecinos_mayor_precision = np.unravel_index(np.argmax(promedio_precisiones), promedio_precisiones.shape)
#    print('Mejor combinacion entre cantidad de columnas y vecinos:\n')
#    print('Cantidad de columnas: ',n_columnas[columnas_mayor_precision],'\nCantidad de vecinos: ',vecinos_mayor_precision+1)
#
#    # hago modelo con esos parametros para ver la matriz de confusion
#    col_mas = funcion_columnas(img_0_1,n_columnas[columnas_mayor_precision])
#    PIXELES = col_mas.iloc[:,1:]
#    DIGITO = col_mas[0]
#    X_train, X_test, Y_train, Y_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
#    model = KNeighborsClassifier(n_neighbors = vecinos_mayor_precision+1)
#    model.fit(X_train, Y_train)
#    predicciones_test = model.predict(X_test)
#    print('\nMatriz de confusion para esos parametros:\n',metrics.confusion_matrix(Y_test,predicciones_test))
#    print('\nPrecision: ',metrics.accuracy_score(Y_test,predicciones_test),'\n')
#
#
#    sns.heatmap(promedio_precisiones, xticklabels=k_vecinos, yticklabels=n_columnas)
#    plt.xlabel('Cantidad de vecinos')
#    plt.ylabel('Cantidad de columnas')
#    plt.title('Precision del modelo para diferentes combinaciones de columnas y vecinos con columnas: '+nombre_funcion)
#    plt.show()
#



def mapa_de_calor(img,funcion_columnas,n_columnas,k,k_folds, nombre_funcion):

    k_vecinos = np.arange(1,k)
    
    resultados_por_foldeo = []
    kf = StratifiedKFold(n_splits=k_folds,shuffle=True) 
    indices = funcion_columnas(img,max(cant_columnas)).columns[1:]
    for train, test in kf.split(PIXELES,DIGITO):
        resultados_por_columna = []
        for n in n_columnas:
            PIXELES_train = img.iloc[train,indices[:n]]
            DIGITO_train = img.iloc[train,0]
            PIXELES_test = img.iloc[test,indices[:n]]
            DIGITO_test = img.iloc[test,0]
            
            resultados_por_vecinos = []
            for k in k_vecinos:
                model = KNeighborsClassifier(n_neighbors = k)
                model.fit(PIXELES_train, DIGITO_train)
                predicciones_test = model.predict(PIXELES_test)
                precision_test = metrics.accuracy_score(DIGITO_test, predicciones_test)
                resultados_por_vecinos.append(precision_test)

        resultados_por_foldeo.append(resultados_por_columna)
                    

    promedio_precisiones = np.mean(resultados_por_foldeo,axis=0)

    mayor_precision = np.max(promedio_precisiones)
    print('\nMayor precision alcanzada: ',mayor_precision)
    print()
    columnas_mayor_precision,vecinos_mayor_precision = np.unravel_index(np.argmax(promedio_precisiones), promedio_precisiones.shape)
    print('Mejor combinacion entre cantidad de columnas y vecinos:\n')
    print('Cantidad de columnas: ',n_columnas[columnas_mayor_precision],'\nCantidad de vecinos: ',vecinos_mayor_precision+1)

    # hago modelo con esos parametros para ver la matriz de confusion
    col_mas = funcion_columnas(img_0_1,n_columnas[columnas_mayor_precision])
    PIXELES = col_mas.iloc[:,1:]
    DIGITO = col_mas[0]
    X_train, X_test, Y_train, Y_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
    model = KNeighborsClassifier(n_neighbors = vecinos_mayor_precision+1)
    model.fit(X_train, Y_train)
    predicciones_test = model.predict(X_test)
    print('\nMatriz de confusion para esos parametros:\n',metrics.confusion_matrix(Y_test,predicciones_test))
    print('\nPrecision: ',metrics.accuracy_score(Y_test,predicciones_test),'\n')


    sns.heatmap(promedio_precisiones, xticklabels=k_vecinos, yticklabels=n_columnas)
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Cantidad de columnas')
    plt.title('Precision del modelo para diferentes combinaciones de columnas y vecinos con columnas: '+nombre_funcion)
    plt.show()


