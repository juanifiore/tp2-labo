'''
Trabajo practico Nº 2, Laboratorio de Datos
Grupo NJL: Falczuk Noelia, Sanes Salazar Luna, Fiore Juan Ignacio

El codigo implementara funciones de la libreria SKLearn para intentar lograr una clasificacion
de imagenes de numeros escritos a mano. Utilizando el csv mnist_desarrollo.csv que contiene un 
conjunto de 60000 imagenes.

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree


#%%

df_img = pd.read_csv('mnist_desarrollo.csv',header=None)
df_img.columns = range(df_img.shape[1])

#%%
#==================================================================================
# EJERCICIO 1
#==================================================================================

print('\n====================================================\nEJERCICIO 1\n====================================================\n')

# 1. Realizar un análisis exploratorio de los datos. Ver, entre otras cosas,
# cantidad de datos, cantidad y tipos de atributos, cantidad de clases de la
# variable de interés (el dígito) y otras características que consideren
# relevantes. ¿Cuáles parecen ser atributos relevantes? ¿Cuáles no? Se
# pueden hacer gráficos para abordar estas preguntas.

#--------------------------------------------------------------------------------
print("Cantidad de datos(imagenes):", len(df_img))
print()
print("Tipos de atributos:", df_img.dtypes[0])
print()
print("Cantidad de clases(digitos):", len(df_img[0].unique()))
print()


# vemos cantidad de imagenes por numero:
cantidad_de_imagenes_por_numero = df_img[0].value_counts().sort_index()
print("=========================\nImagenes por digito\n=========================")
print("Las cantidades por digito son: ")
print(cantidad_de_imagenes_por_numero)
print()

# proporciones
proporciones_de_imagenes_por_numero = df_img[0].value_counts(normalize=True).sort_index()
print("=========================\nProporciones de imagenes por digito\n=========================")
print("Las proporciones por digito son: ")
print(proporciones_de_imagenes_por_numero)
print()

# graficamos la cantidad en un barplot con el promedio
#sns.barplot(x='numero',y='cantidad',data=pd.DataFrame({'numero': cantidad_de_imagenes_por_numero.index, 'cantidad': cantidad_de_imagenes_por_numero.values}))
#plt.axhline(cantidad_de_imagenes_por_numero.mean(), color='red', linestyle='--')
#plt.show()
#plt.close()

#--------------------------------------------------------------------------------


#%%
#==================================================================================
# EJERCICIO 2
#==================================================================================

print('\n====================================================\nEJERCICIO 2\n====================================================\n')

# 2. Construir un dataframe con el subconjunto que contiene solamente los
# dígitos 0 y 1.

#--------------------------------------------------------------------------------

img_0_1 = df_img[(df_img[0] == 0) | (df_img[0] == 1)]

#--------------------------------------------------------------------------------

#%%
#==================================================================================
# EJERCICIO 3
#==================================================================================

print('\n====================================================\nEJERCICIO 3\n====================================================\n')

# 3. Para este subconjunto de datos, ver cuántas muestras se tienen y
# determinar si está balanceado entre las clases.

#--------------------------------------------------------------------------------

# vemos cantidad de imagenes por numero:
cantidad_de_imagenes_0_1 = img_0_1[0].value_counts().sort_index()
print("=========================\nImagenes por digito\n=========================")
print("Las cantidades por digito son: ")
print(cantidad_de_imagenes_0_1)
print()

#cantidad total
print('Cantidad total:')
print(cantidad_de_imagenes_0_1.sum())
print()

# proporciones
proporciones_de_imagenes_0_1 = img_0_1[0].value_counts(normalize=True).sort_index()
print("=========================\nProporciones de imagenes por digito\n=========================")
print("Las proporciones por digito son: ")
print(proporciones_de_imagenes_0_1*100)
print()

# graficamos la cantidad en un barplot con el promedio
#sns.barplot(x='numero',y='cantidad',data=pd.DataFrame({'numero': cantidad_de_imagenes_0_1.index, 'cantidad': cantidad_de_imagenes_0_1.values}))
#plt.axhline(cantidad_de_imagenes_0_1.mean(), color='red', linestyle='--')
#plt.show()
#plt.close()


#--------------------------------------------------------------------------------


#%%
#==================================================================================
# EJERCICIO 4
#==================================================================================

print('\n====================================================\nEJERCICIO 4\n====================================================\n')

# 4. Ajustar un modelo de knn considerando pocos atributos, por ejemplo 3.
# Probar con distintos conjuntos de 3 atributos y comparar resultados.
# Analizar utilizando otras cantidades de atributos.

#--------------------------------------------------------------------------------

# armaremos nuevos dataframe que contengan solo n atributos (pixeles o columnas)
# primero n al azar, luego n equidistantes, y luego las n columnas que contengan mayor cantidad de datos distintos de cero
# tambien para probar, viendo que el elemento que mas aparece es el 253, tomo las n columnas que mas contengan a ese numero
# y otro con el 255, que es el color mas fuerte de la imagen

# estos son los 6 elementos que mas aparecen en el df img_0_1 y la cantidad de apariciones
#0      8219442
#253     345742
#252     138242
#254     122271
#255      65771


# al azar
def n_col_al_azar(img,n):
    indices_al_azar = np.random.randint(1, 785, size=n)
    col_al_azar = img.iloc[:,np.insert(indices_al_azar,0,0)]
    return col_al_azar

# menos cantidad de ceros
def n_col_menos_ceros(img,n):
    cantidades_de_ceros = (img.iloc[:,1:]==0).sum()
    indices_n_columnas_menos_ceros = cantidades_de_ceros.sort_values().index[:n]
    col_menos_ceros = img[np.insert(indices_n_columnas_menos_ceros,0,0)]    # le agrego la columna 0
    return col_menos_ceros

# equidistantes (n columnas que parten las imagenes en (n+1) partes iguales)
def n_col_equi_dist(img,n):
    indices_equi_dist = np.arange(784/(n+1), 784, np.ceil(784/(n+1)),dtype=int)
    col_equi_dist = img[np.insert(indices_equi_dist,0,0)]   # le agrego la columna 0 
    return col_equi_dist

# columnas que mas tienen el 253
def n_col_mas_253(img,n):
    cantidades_de_253 = (img.iloc[:,1:]==253).sum()
    indices_n_columnas_mas_253 = cantidades_de_253.sort_values(ascending=False).index[:n]
    col_mas_253 = img[np.insert(indices_n_columnas_mas_253,0,0)]    # le agrego la columna 0
    return col_mas_253

# columnas que mas tienen el 255
def n_col_mas_255(img,n):
    cantidades_de_255 = (img.iloc[:,1:]==255).sum()
    indices_n_columnas_mas_255 = cantidades_de_255.sort_values(ascending=False).index[:n]
    col_mas_255 = img[np.insert(indices_n_columnas_mas_255,0,0)]    # le agrego la columna 0
    return col_mas_255

def n_col_mas_var(img,n):
    varianzas = np.var(img, axis=0)
    indices_n_columnas_mas_var = varianzas.sort_values(ascending=False).index[:n]
    col_mas_var = img[np.insert(indices_n_columnas_mas_var,0,0)]    # le agrego la columna 0
    return col_mas_var 

def n_col_mas_norma2(img,n):
    norma2 = np.linalg.norm(img, axis=0)
    indices_n_columnas_mas_norma2 = np.argsort(np.linalg.norm(img_0_1,axis=0))[-n:]
    col_mas_norma2 = img[np.insert(indices_n_columnas_mas_norma2,0,0)]    # le agrego la columna 0
    return col_mas_norma2 



#--------------------------------------------------------------------------------
#%%


n=3

col_al_azar = n_col_al_azar(img_0_1,n)
col_menos_ceros = n_col_menos_ceros(img_0_1,n)
col_equi_dist = n_col_equi_dist(img_0_1,n)
col_mas_253 = n_col_mas_253(img_0_1,n)
col_mas_255 = n_col_mas_255(img_0_1,n)
col_mas_var = n_col_mas_var(img_0_1,n)
col_mas_norma2 = n_col_mas_norma2(img_0_1,n)

k = 5

print('----------\nNumero de columnas: ',n,'\nNumero de vecinos: ',k,'\n-----------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de menos ceros\n=========================')

PIXELES = col_menos_ceros.iloc[:,1:]
DIGITO = col_menos_ceros[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de mas 253 \n=========================')

PIXELES = col_mas_253.iloc[:,1:]
DIGITO = col_mas_253[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de mas 255 \n=========================')

PIXELES = col_mas_255.iloc[:,1:]
DIGITO = col_mas_255[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas al azar\n=========================')

PIXELES = col_al_azar.iloc[:,1:]
DIGITO = col_al_azar[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas equidistantes\n=========================')

PIXELES = col_equi_dist.iloc[:,1:]
DIGITO = col_equi_dist[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de mas varianza\n=========================')

PIXELES = col_mas_var.iloc[:,1:]
DIGITO = col_mas_var[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de mas norma2\n=========================')

PIXELES = col_mas_norma2.iloc[:,1:]
DIGITO = col_mas_norma2[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo qué clases les asigna el modelo a mis datos
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')




#--------------------------------------------------------------------------------

#%%
#==================================================================================
# EJERCICIO 5
#==================================================================================

print('\n====================================================\nEJERCICIO 5\n====================================================\n')

# 5. Para comparar modelos, utilizar validación cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el análisis
# de los resultados, tener en cuenta las medidas de evaluación (por ejemplo,
# la exactitud) y la cantidad de atributos.

#--------------------------------------------------------------------------------

def mejor_seleccion_columnas(imagenes,cant_columnas,tuplas_funciones,k_vecinos,i_rep):

    valores_k = range(1,k_vecinos)

    for funcion in tuplas_funciones:
        columnas_selec = funcion[1](imagenes,cant_columnas)
        PIXELES = columnas_selec.iloc[:,1:]
        DIGITO = columnas_selec[0]
        resultados_test = np.zeros((i_rep, len(valores_k)))
        for i in range(i_rep):
            X_train, X_test, Y_train, Y_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
            for k in valores_k:
                model = KNeighborsClassifier(n_neighbors = k)
                model.fit(X_train, Y_train)
                predicciones_test = model.predict(X_test)
                precision_test = metrics.accuracy_score(Y_test, predicciones_test)
                resultados_test[i, k-1] = precision_test

        promedios_test = np.mean(resultados_test, axis = 0)
            
        plt.plot(valores_k, promedios_test, label = funcion[0])
    plt.legend()
    plt.title('Exactitud del modelo de knn con ' + str(n) + ' columnas')
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Exactitud (accuracy)')
    plt.show()


tuplas_funciones = [('Columnas  mas 255',n_col_mas_255),('Columnas menos ceros',n_col_menos_ceros),('Columnas mas varianza',n_col_mas_var)]

#mejor_seleccion_columnas(img_0_1,3,tuplas_funciones,10,3)

print('mejor_seleccion_columnas(imagenes,cant_columnas,tuplas_funciones,k_vecinos,i_rep)')

#--------------------------------------------------------------------------------

# viendo que las mejores columnas son las de mas cantidad de pixeles de 255, hago analisis sobre esto en busca de la mejor clasificacion con knn
# buscaremos la mejor combinacion entre cantidad de columnas y cantidad de vecinos y veremos las relaciones a partir de un mapa de calor
# a la funcion se le pasa el DF de las imagenes, la lista de cantidades de columnas, la cantidad de vecinos, la cantidad de iteraciones de testeo

def mejor_modelo_255(imagenes,n_columnas,k,i_rep):
    k_vecinos = np.arange(1,k)
    
    resultados_test_255 = np.zeros((i_rep,len(n_columnas), len(k_vecinos)))
    
    j = -1  # lo voy a usar para asignar en la posicion j de la matriz
    for n in n_columnas:
        j += 1
        col_mas_var = n_col_mas_var(imagenes,n)
        PIXELES_var = col_mas_var.iloc[:,1:]
        DIGITO_var = col_mas_var[0]
        for i in range(i_rep):
            X_train_var, X_test_var, Y_train_var, Y_test_var = train_test_split(PIXELES_var,DIGITO_var, test_size = 0.3)
            for k in k_vecinos:
                model_var = KNeighborsClassifier(n_neighbors = k)
                model_var.fit(X_train_var, Y_train_var)
                predicciones_test_var = model_var.predict(X_test_var)
                precision_test_var = metrics.accuracy_score(Y_test_var, predicciones_test_var)
                resultados_test_var[i,j,k-1] = precision_test_var

    promedio_precisiones = np.mean(resultados_test_var,axis=0)

    mayor_precision = np.max(promedio_precisiones)
    print('\nMayor precision alcanzada: ',mayor_precision)
    print()
    columnas_mayor_precision,vecinos_mayor_precision = np.unravel_index(np.argmax(promedio_precisiones), promedio_precisiones.shape)
    print('Mejor combinacion entre cantidad de columnas y vecinos:\n')
    print('Cantidad de columnas: ',n_columnas[columnas_mayor_precision],'\nCantidad de vecinos: ',vecinos_mayor_precision+1)

    # hago modelo con esos parametros para ver la matriz de confusion
    col_mas_var = n_col_mas_var(img_0_1,n_columnas[columnas_mayor_precision])
    PIXELES_var = col_mas_var.iloc[:,1:]
    DIGITO_var = col_mas_var[0]
    X_train_var, X_test_var, Y_train_var, Y_test_var = train_test_split(PIXELES_var,DIGITO_var, test_size = 0.3)
    model_var = KNeighborsClassifier(n_neighbors = vecinos_mayor_precision+1)
    model_var.fit(X_train_var, Y_train_var)
    predicciones_test_var = model_var.predict(X_test_var)
    print('\nMatriz de confusion para esos parametros:\n',metrics.confusion_matrix(Y_test_var,predicciones_test_var))
    print('\nPrecision: ',metrics.accuracy_score(Y_test_var,predicciones_test_var),'\n')


    sns.heatmap(promedio_precisiones, xticklabels=k_vecinos, yticklabels=n_columnas)
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Cantidad de columnas')
    plt.title('Precision del modelo para diferentes combinaciones de columnas y vecinos')
    plt.show()

#    return resultados_test_255

print('\nFuncion mejor_modelo_255(img_0_1,columnas,k,i)')

#-----------------------------------------------------------
# funcion de modelo KNN, se le pasa como parametro el DF de las imagenes, las cantidad de columnas,
# la cantidad de vecinos, y la funcion con la que se seleccionan la cantidad de columnas (azar, menos ceros, mas cantidad de 255,...)
# la cantidad de columnas es fija

def knn_255(img,cant_columnas,k_vecinos,funcion_columnas):
    img_columnas_selec = funcion_columnas(img,cant_columnas)
    PIXELES = img_columnas_selec.iloc[:,1:]
    DIGITO = img_columnas_selec[0]
    PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
    model = KNeighborsClassifier(n_neighbors = k_vecinos)
    model.fit(PIXELES_train, DIGITO_train)
    PREDICCIONES = model.predict(PIXELES_test)
    print('----------------\nPrecision: \n')
    print(metrics.accuracy_score(DIGITO_test, PREDICCIONES))
    print('----------------\nMatriz de confusion\n')
    print(metrics.confusion_matrix(DIGITO_test, PREDICCIONES))
    print('----------------\n')
    return model


# funcion que grafica la relacion entre la cantidad de columnas y la precision, pasandole tambien
# como parametro a la funcion con la que se seleccionan las columnas (azar, menos ceros, mas cantidad de 255,...)
# las columnas se pasan en forma de lista, cada elemento es una cantidad de columnas a analizar
# se pasa una lista con la cantidad de vecinos a probar

def relacion_cant_columnas_precision(img,cant_columnas,funcion_columnas,k_vecinos):
    for k in k_vecinos:
        precisiones = [] 
        for n in cant_columnas:
            img_columnas_selec = funcion_columnas(img,n)
            PIXELES = img_columnas_selec.iloc[:,1:]
            DIGITO = img_columnas_selec[0]
            PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
            model = KNeighborsClassifier(n_neighbors = k)
            model.fit(PIXELES_train, DIGITO_train)
            PREDICCIONES = model.predict(PIXELES_test)
            precision = metrics.accuracy_score(DIGITO_test, PREDICCIONES)
            precisiones.append(precision)
        plt.scatter(cant_columnas,precisiones,label=str(k)+' vecinos')
    plt.title('Relacion entre cantidad de columnas, vecinos y precision')
    plt.xlabel('Cantidad de columnas')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()




#-----------------------------------------------------------


#%%
#==================================================================================
# EJERCICIO 6
#==================================================================================

print('\n====================================================\nEJERCICIO 6\n====================================================\n')

# 6. Trabajar nuevamente con el dataset de todos los dígitos. Ajustar un
# modelo de árbol de decisión. Analizar distintas profundidades.

#--------------------------------------------------------------------------------


def arbol(df,profundidades):
    PIXELES = df.iloc[:,1:]
    DIGITO = df[0]
    profundidad_mas_eficiente = 0
    mejor_precision = 0
    matriz = np.zeros((10,10))
    for p in profundidades:
        arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= p)
        arbol = arbol.fit(PIXELES, DIGITO)

        PREDICCIONES = arbol.predict(PIXELES) 

        precision = metrics.accuracy_score(DIGITO,PREDICCIONES)
        
        if precision > mejor_precision:
            mejor_precision = precision
            profundidad_mas_eficiente = p
            matriz = metrics.confusion_matrix(DIGITO,PREDICCIONES)

    print('--------------------------\n')
    print('Mejor profundidad: ',profundidad_mas_eficiente,'\n')
    print('Precision: ',mejor_precision,'\n')
    print('Matriz de confusion:\n')
    print(matriz)
    print('--------------------------\n')

print('\nFuncion arbol(df,profundidades)\n')


#--------------------------------------------------------------------------------


#%%
#==================================================================================
# EJERCICIO 7
#==================================================================================

print('\n====================================================\nEJERCICIO 7\n====================================================\n')

# 7. Para comparar y seleccionar los árboles de decisión, utilizar validación
# cruzada con k-folding.

#--------------------------------------------------------------------------------

def arbol(df,profundidades):
    PIXELES = df.iloc[:,1:]
    DIGITO = df[0]
    profundidad_mas_eficiente = 0
    mejor_precision = 0
    matriz = np.zeros((10,10))
    for p in profundidades:
        PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
        arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= p)
        arbol = arbol.fit(PIXELES_train, DIGITO_train)

        PREDICCIONES = arbol.predict(PIXELES_test) 

        precision = metrics.accuracy_score(DIGITO_test, PREDICCIONES)
        
        if precision > mejor_precision:
            mejor_precision = precision
            profundidad_mas_eficiente = p
            matriz = metrics.confusion_matrix(DIGITO_test,PREDICCIONES)

    print('--------------------------\n')
    print('Mejor profundidad: ',profundidad_mas_eficiente,'\n')
    print('Precision: ',mejor_precision,'\n')
    print('Matriz de confusion:\n')
    print(matriz)
    print('--------------------------\n')

def relacion_cant_columnas_precision_arbol(img,cant_columnas,funcion_columnas,profundidad,nombre_funcion):
    for p in profundidad:
        precisiones = [] 
        for n in cant_columnas:
            img_columnas_selec = funcion_columnas(img,n)
            PIXELES = img_columnas_selec.iloc[:,1:]
            DIGITO = img_columnas_selec[0]
            PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
            arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= p)
            arbol = arbol.fit(PIXELES_train, DIGITO_train)
            PREDICCIONES = arbol.predict(PIXELES_test) 
            precision = metrics.accuracy_score(DIGITO_test, PREDICCIONES)
            precisiones.append(precision)

        plt.scatter(cant_columnas,precisiones,label='Profundidad: '+str(p))
    plt.title('Relacion entre cantidad de columnas, profundidad y precision, '+nombre_funcion)
    plt.xlabel('Cantidad de columnas')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

print('Funcion relacion_cant_columnas_precision_arbol(img,cant_columnas,funcion_columnas,profundidad,nombre_funcion)')


def mejor_seleccion_columnas_arbol(imagenes,cant_columnas,tuplas_funciones,profundidades,i_rep):
    for funcion in tuplas_funciones:
        columnas_selec = funcion[1](imagenes,cant_columnas)
        PIXELES = columnas_selec.iloc[:,1:]
        DIGITO = columnas_selec[0]
        resultados_test = np.zeros((i_rep, len(profundidades)))
        for i in range(i_rep):
            PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
            for p in profundidades:
                arbol = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= p)
                arbol = arbol.fit(PIXELES_train, DIGITO_train)
                PREDICCIONES = arbol.predict(PIXELES_test) 
                precision_test = metrics.accuracy_score(DIGITO_test, PREDICCIONES)
                resultados_test[i, p-1] = precision_test

        promedios_test = np.mean(resultados_test, axis = 0)
            
        plt.plot(profundidades, promedios_test, label = funcion[0])
    plt.legend()
    plt.title('Exactitud del modelo de knn con ' + str(cant_columnas) + ' columnas')
    plt.xlabel('Profundidad')
    plt.ylabel('Exactitud (accuracy)')
    plt.show()

tuplas_funciones = [('Columnas  mas 255',n_col_mas_255),('Columnas menos ceros',n_col_menos_ceros),('Columnas mas varianza',n_col_mas_var)]

mejor_seleccion_columnas_arbol(img_0_1,3,tuplas_funciones,np.arange(1,15),3)

print('\nFuncion: mejor_seleccion_columnas_arbol(imagenes,cant_columnas,tuplas_funciones,profundidades,i_rep)')






#--------------------------------------------------------------------------------

