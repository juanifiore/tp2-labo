def mejor_seleccion_columnas(imagenes,cant_columnas,tuplas_funciones,k_vecinos,i_rep):

    valores_k = rang(1,k_vecinos)

    for funcion in tuplas_funciones:
        columnas_selec = funcion[1](imagenes,cant_columnas)
        PIXELES = columnas_selec.iloc[:,1:]
        DIGITO = columnas_selec[0]
        resultados_test = np.zeros((Nrep, len(valores_k)))
        for i in range(Nrep):
            X_train, X_test, Y_train, Y_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
            for k in valores_k:
                model = KNeighborsClassifier(n_neighbors = k)
                model.fit(X_train, Y_train)
                predicciones_test = model.predict(X_test)
                precision_test = metrics.accuracy_score(Y_test, predicciones_test)
                resultados_test[i, k-1] = precision_test

        promedios_test = np.mean(resultados_test, axis = 0)
            
        plt.plot(valores_k, promedios_test_equi, label = funcion[0])
    plt.legend()
    plt.title('Exactitud del modelo de knn con ' + str(n) + ' columnas')
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Exactitud (accuracy)')
    plt.show()




#n=3
#
## armo los nuevos dataframe con n columnas 
## el de columnas al azar lo genero dentro del ciclo, para poder analizar mejor el "azar".
## se generara un nuevo DF de columnas al azar por cada iteracion dada por 'Nrep'.
#
#
#col_menos_ceros = n_col_menos_ceros(img_0_1,n)
#col_equi_dist = n_col_equi_dist(img_0_1,n)
#col_mas_253 = n_col_mas_253(img_0_1,n)
#col_mas_255 = n_col_mas_255(img_0_1,n)
#col_mas_var = n_col_mas_var(img_0_1,n)
#col_mas_norma2 = n_col_mas_norma2(img_0_1,n)
#
#PIXELES_equi = col_equi_dist.iloc[:,1:]
#DIGITO_equi = col_equi_dist[0]
#PIXELES_ceros = col_menos_ceros.iloc[:,1:]
#DIGITO_ceros = col_menos_ceros[0]
#PIXELES_253 = col_mas_253.iloc[:,1:]
#DIGITO_253 = col_mas_253[0]
#PIXELES_255 = col_mas_255.iloc[:,1:]
#DIGITO_255 = col_mas_255[0]
#PIXELES_var = col_mas_var.iloc[:,1:]
#DIGITO_var = col_mas_var[0]
#PIXELES_norma2 = col_mas_norma2.iloc[:,1:]
#DIGITO_norma2 = col_mas_norma2[0]
#
#
#Nrep = 0
#valores_k = range(1, 10)
#
##resultados_test_azar = np.zeros((Nrep, len(valores_k)))
##resultados_train_azar = np.zeros((Nrep, len(valores_k)))
#resultados_test_equi = np.zeros((Nrep, len(valores_k)))
##resultados_train_equi = np.zeros((Nrep, len(valores_k)))
#resultados_test_ceros = np.zeros((Nrep, len(valores_k)))
##resultados_train_ceros = np.zeros((Nrep, len(valores_k)))
#resultados_test_253 = np.zeros((Nrep, len(valores_k)))
##resultados_train_253 = np.zeros((Nrep, len(valores_k)))
#resultados_test_255 = np.zeros((Nrep, len(valores_k)))
##resultados_train_255 = np.zeros((Nrep, len(valores_k)))
#resultados_test_var = np.zeros((Nrep, len(valores_k)))
#resultados_test_norma2 = np.zeros((Nrep, len(valores_k)))
#
#
#for i in range(Nrep):
##    col_al_azar = n_col_al_azar(img_0_1,n)
##    PIXELES_azar = col_al_azar.iloc[:,1:]
##    DIGITO_azar = col_al_azar[0]
##    X_train_azar, X_test_azar, Y_train_azar, Y_test_azar = train_test_split(PIXELES_azar,DIGITO_azar, test_size = 0.3)
#    X_train_equi, X_test_equi, Y_train_equi, Y_test_equi = train_test_split(PIXELES_equi,DIGITO_equi, test_size = 0.3)
#    X_train_ceros, X_test_ceros, Y_train_ceros, Y_test_ceros = train_test_split(PIXELES_ceros,DIGITO_ceros, test_size = 0.3)
#    X_train_253, X_test_253, Y_train_253, Y_test_253 = train_test_split(PIXELES_253,DIGITO_253, test_size = 0.3)
#    X_train_255, X_test_255, Y_train_255, Y_test_255 = train_test_split(PIXELES_255,DIGITO_255, test_size = 0.3)
#    X_train_var, X_test_var, Y_train_var, Y_test_var = train_test_split(PIXELES_var,DIGITO_var, test_size = 0.3)
#    X_train_norma2, X_test_norma2, Y_train_norma2, Y_test_norma2 = train_test_split(PIXELES_norma2,DIGITO_norma2, test_size = 0.3)
#    for k in valores_k:
##        # modelo para columnas al azar
##        model_azar = KNeighborsClassifier(n_neighbors = k)
##        model_azar.fit(X_train_azar, Y_train_azar)
##        predicciones_test_azar = model_azar.predict(X_test_azar)
###        predicciones_train_azar = model.predict(X_train)
##        precision_test_azar = metrics.accuracy_score(Y_test_azar, predicciones_test_azar)
###        precision_train_azar = metrics.accuracy_score(Y_train_azar, predicciones_train_azar)
##        resultados_test_azar[i, k-1] = precision_test_azar
###        resultados_train_azar[i, k-1] = precision_train_azar
#
#        #mmodelo para columnas equidistantes
#        model_equi = KNeighborsClassifier(n_neighbors = k)
#        model_equi.fit(X_train_equi, Y_train_equi)
#        predicciones_test_equi = model_equi.predict(X_test_equi)
##        predicciones_train_equi = model.predict(X_train)
#        precision_test_equi = metrics.accuracy_score(Y_test_equi, predicciones_test_equi)
##        precision_train_equi = metrics.accuracy_score(Y_train_equi, predicciones_train_equi)
#        resultados_test_equi[i, k-1] = precision_test_equi
##        resultados_train_equi[i, k-1] = precision_train_equi
#
#        #modelo para columnas con menos ceros
#        model_ceros = KNeighborsClassifier(n_neighbors = k)
#        model_ceros.fit(X_train_ceros, Y_train_ceros)
#        predicciones_test_ceros = model_ceros.predict(X_test_ceros)
##        predicciones_train_ceros = model.predict(X_train)
#        precision_test_ceros = metrics.accuracy_score(Y_test_ceros, predicciones_test_ceros)
##        precision_train_ceros = metrics.accuracy_score(Y_train_ceros, predicciones_train_ceros)
#        resultados_test_ceros[i, k-1] = precision_test_ceros
##        resultados_train_ceros[i, k-1] = precision_train_ceros
#
#        #modelo para columnas con mas 253
#        model_253 = KNeighborsClassifier(n_neighbors = k)
#        model_253.fit(X_train_253, Y_train_253)
#        predicciones_test_253 = model_253.predict(X_test_253)
##        predicciones_train_253 = model.predict(X_train)
#        precision_test_253 = metrics.accuracy_score(Y_test_253, predicciones_test_253)
##        precision_train_253 = metrics.accuracy_score(Y_train_253, predicciones_train_253)
#        resultados_test_253[i, k-1] = precision_test_253
##        resultados_train_253[i, k-1] = precision_train_253
#
#        #modelo para columnas con mas 255
#        model_255 = KNeighborsClassifier(n_neighbors = k)
#        model_255.fit(X_train_255, Y_train_255)
#        predicciones_test_255 = model_255.predict(X_test_255)
##        predicciones_train_255 = model.predict(X_train)
#        precision_test_255 = metrics.accuracy_score(Y_test_255, predicciones_test_255)
##        precision_train_255 = metrics.accuracy_score(Y_train_255, predicciones_train_255)
#        resultados_test_255[i, k-1] = precision_test_255
##        resultados_train_255[i, k-1] = precision_train_255
#
#        #modelo para columnas con mas var
#        model_var = KNeighborsClassifier(n_neighbors = k)
#        model_var.fit(X_train_var, Y_train_var)
#        predicciones_test_var = model_var.predict(X_test_var)
##        predicciones_train_var = model.predict(X_train)
#        precision_test_var = metrics.accuracy_score(Y_test_var, predicciones_test_var)
##        precision_train_var = metrics.accuracy_score(Y_train_var, predicciones_train_var)
#        resultados_test_var[i, k-1] = precision_test_var
##        resultados_train_var[i, k-1] = precision_train_var
#        
#        #modelo para columnas con mas norma2
#        model_norma2 = KNeighborsClassifier(n_neighbors = k)
#        model_norma2.fit(X_train_norma2, Y_train_norma2)
#        predicciones_test_norma2 = model_norma2.predict(X_test_norma2)
##        predicciones_train_norma2 = model.predict(X_train)
#        precision_test_norma2 = metrics.accuracy_score(Y_test_norma2, predicciones_test_norma2)
##        precision_train_norma2 = metrics.accuracy_score(Y_train_norma2, predicciones_train_norma2)
#        resultados_test_norma2[i, k-1] = precision_test_norma2
##        resultados_train_norma2[i, k-1] = precision_train_norma2
#
#
#
#
##promedios_train_azar = np.mean(resultados_train_azar, axis = 0)
##promedios_test_azar = np.mean(resultados_test_azar, axis = 0)
##promedios_train_equi = np.mean(resultados_train_equi, axis = 0)
#promedios_test_equi = np.mean(resultados_test_equi, axis = 0)
##promedios_train_ceros = np.mean(resultados_train_ceros, axis = 0)
#promedios_test_ceros = np.mean(resultados_test_ceros, axis = 0)
##promedios_train_253 = np.mean(resultados_train_253, axis = 0)
#promedios_test_253 = np.mean(resultados_test_253, axis = 0)
##promedios_train_255 = np.mean(resultados_train_255, axis = 0)
#promedios_test_255 = np.mean(resultados_test_255, axis = 0)
#promedios_test_var = np.mean(resultados_test_var, axis = 0)
#promedios_test_norma2 = np.mean(resultados_test_norma2, axis = 0)
##%%
#
##plt.plot(valores_k, promedios_test_azar, label = 'Columnas al azar')
#plt.plot(valores_k, promedios_test_equi, label = 'Columnas equidistantes')
#plt.plot(valores_k, promedios_test_ceros, label = 'Columnas con menos ceros')
#plt.plot(valores_k, promedios_test_253, label = 'Columnas con mas 253')
#plt.plot(valores_k, promedios_test_255, label = 'Columnas con mas 255')
#plt.plot(valores_k, promedios_test_var, label = 'Columnas con mas var')
#plt.plot(valores_k, promedios_test_norma2, label = 'Columnas con mas norma2')
#plt.legend()
#plt.title('Exactitud del modelo de knn con ' + str(n) + ' columnas')
#plt.xlabel('Cantidad de vecinos')
#plt.ylabel('Exactitud (accuracy)')
#plt.show()
#


