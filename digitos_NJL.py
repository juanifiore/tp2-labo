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



#--------------------------------------------------------------------------------
#%%


n=3

col_al_azar = n_col_al_azar(img_0_1,n)
col_menos_ceros = n_col_menos_ceros(img_0_1,n)
col_equi_dist = n_col_equi_dist(img_0_1,n)
col_mas_253 = n_col_mas_253(img_0_1,n)
col_mas_255 = n_col_mas_255(img_0_1,n)

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


n=3

# armo los nuevos dataframe con n columnas 
# el de columnas al azar lo genero dentro del ciclo, para poder analizar mejor el "azar".
# se generara un nuevo DF de columnas al azar por cada iteracion dada por 'Nrep'.

col_menos_ceros = n_col_menos_ceros(img_0_1,n)
col_equi_dist = n_col_equi_dist(img_0_1,n)
col_mas_253 = n_col_mas_253(img_0_1,n)
col_mas_255 = n_col_mas_255(img_0_1,n)

PIXELES_equi = col_equi_dist.iloc[:,1:]
DIGITO_equi = col_equi_dist[0]
PIXELES_ceros = col_menos_ceros.iloc[:,1:]
DIGITO_ceros = col_menos_ceros[0]
PIXELES_253 = col_mas_253.iloc[:,1:]
DIGITO_253 = col_mas_253[0]
PIXELES_255 = col_mas_255.iloc[:,1:]
DIGITO_255 = col_mas_255[0]


Nrep = 0
valores_k = range(1, 20)

#resultados_test_azar = np.zeros((Nrep, len(valores_k)))
#resultados_train_azar = np.zeros((Nrep, len(valores_k)))
resultados_test_equi = np.zeros((Nrep, len(valores_k)))
#resultados_train_equi = np.zeros((Nrep, len(valores_k)))
resultados_test_ceros = np.zeros((Nrep, len(valores_k)))
#resultados_train_ceros = np.zeros((Nrep, len(valores_k)))
resultados_test_253 = np.zeros((Nrep, len(valores_k)))
#resultados_train_253 = np.zeros((Nrep, len(valores_k)))
resultados_test_255 = np.zeros((Nrep, len(valores_k)))
#resultados_train_255 = np.zeros((Nrep, len(valores_k)))


for i in range(Nrep):
#    col_al_azar = n_col_al_azar(img_0_1,n)
#    PIXELES_azar = col_al_azar.iloc[:,1:]
#    DIGITO_azar = col_al_azar[0]
#    X_train_azar, X_test_azar, Y_train_azar, Y_test_azar = train_test_split(PIXELES_azar,DIGITO_azar, test_size = 0.3)
    X_train_equi, X_test_equi, Y_train_equi, Y_test_equi = train_test_split(PIXELES_equi,DIGITO_equi, test_size = 0.3)
    X_train_ceros, X_test_ceros, Y_train_ceros, Y_test_ceros = train_test_split(PIXELES_ceros,DIGITO_ceros, test_size = 0.3)
    X_train_253, X_test_253, Y_train_253, Y_test_253 = train_test_split(PIXELES_253,DIGITO_253, test_size = 0.3)
    X_train_255, X_test_255, Y_train_255, Y_test_255 = train_test_split(PIXELES_255,DIGITO_255, test_size = 0.3)
    for k in valores_k:
#        # modelo para columnas al azar
#        model_azar = KNeighborsClassifier(n_neighbors = k)
#        model_azar.fit(X_train_azar, Y_train_azar)
#        predicciones_test_azar = model_azar.predict(X_test_azar)
##        predicciones_train_azar = model.predict(X_train)
#        precision_test_azar = metrics.accuracy_score(Y_test_azar, predicciones_test_azar)
##        precision_train_azar = metrics.accuracy_score(Y_train_azar, predicciones_train_azar)
#        resultados_test_azar[i, k-1] = precision_test_azar
##        resultados_train_azar[i, k-1] = precision_train_azar

        #mmodelo para columnas equidistantes
        model_equi = KNeighborsClassifier(n_neighbors = k)
        model_equi.fit(X_train_equi, Y_train_equi)
        predicciones_test_equi = model_equi.predict(X_test_equi)
#        predicciones_train_equi = model.predict(X_train)
        precision_test_equi = metrics.accuracy_score(Y_test_equi, predicciones_test_equi)
#        precision_train_equi = metrics.accuracy_score(Y_train_equi, predicciones_train_equi)
        resultados_test_equi[i, k-1] = precision_test_equi
#        resultados_train_equi[i, k-1] = precision_train_equi

        #modelo para columnas con menos ceros
        model_ceros = KNeighborsClassifier(n_neighbors = k)
        model_ceros.fit(X_train_ceros, Y_train_ceros)
        predicciones_test_ceros = model_ceros.predict(X_test_ceros)
#        predicciones_train_ceros = model.predict(X_train)
        precision_test_ceros = metrics.accuracy_score(Y_test_ceros, predicciones_test_ceros)
#        precision_train_ceros = metrics.accuracy_score(Y_train_ceros, predicciones_train_ceros)
        resultados_test_ceros[i, k-1] = precision_test_ceros
#        resultados_train_ceros[i, k-1] = precision_train_ceros

        #modelo para columnas con mas 253
        model_253 = KNeighborsClassifier(n_neighbors = k)
        model_253.fit(X_train_253, Y_train_253)
        predicciones_test_253 = model_253.predict(X_test_253)
#        predicciones_train_253 = model.predict(X_train)
        precision_test_253 = metrics.accuracy_score(Y_test_253, predicciones_test_253)
#        precision_train_253 = metrics.accuracy_score(Y_train_253, predicciones_train_253)
        resultados_test_253[i, k-1] = precision_test_253
#        resultados_train_253[i, k-1] = precision_train_253

        #modelo para columnas con mas 255
        model_255 = KNeighborsClassifier(n_neighbors = k)
        model_255.fit(X_train_255, Y_train_255)
        predicciones_test_255 = model_255.predict(X_test_255)
#        predicciones_train_255 = model.predict(X_train)
        precision_test_255 = metrics.accuracy_score(Y_test_255, predicciones_test_255)
#        precision_train_255 = metrics.accuracy_score(Y_train_255, predicciones_train_255)
        resultados_test_255[i, k-1] = precision_test_255
#        resultados_train_255[i, k-1] = precision_train_255

#%%

#promedios_train_azar = np.mean(resultados_train_azar, axis = 0)
#promedios_test_azar = np.mean(resultados_test_azar, axis = 0)
#promedios_train_equi = np.mean(resultados_train_equi, axis = 0)
promedios_test_equi = np.mean(resultados_test_equi, axis = 0)
#promedios_train_ceros = np.mean(resultados_train_ceros, axis = 0)
promedios_test_ceros = np.mean(resultados_test_ceros, axis = 0)
#promedios_train_253 = np.mean(resultados_train_253, axis = 0)
promedios_test_253 = np.mean(resultados_test_253, axis = 0)
#promedios_train_255 = np.mean(resultados_train_255, axis = 0)
promedios_test_255 = np.mean(resultados_test_255, axis = 0)
#%%

#plt.plot(valores_k, promedios_test_azar, label = 'Columnas al azar')
plt.plot(valores_k, promedios_test_equi, label = 'Columnas equidistantes')
plt.plot(valores_k, promedios_test_ceros, label = 'Columnas con menos ceros')
plt.plot(valores_k, promedios_test_253, label = 'Columnas con mas 253')
plt.plot(valores_k, promedios_test_255, label = 'Columnas con mas 255')
plt.legend()
plt.title('Exactitud del modelo de knn con ' + str(n) + ' columnas')
plt.xlabel('Cantidad de vecinos')
plt.ylabel('Exactitud (accuracy)')
plt.show()

#--------------------------------------------------------------------------------

# viendo que las mejores columnas son las de mas cantidad de pixeles de 255, hago analisis sobre esto en busca de la mejor clasificacion con knn
# buscaremos la mejor combinacion entre cantidad de columnas y cantidad de vecinos y veremos las relaciones a partir de un mapa de calor

def mejor_modelo_255(img_0_1,columnas,k,i):
    n_columnas = columnas
    k_vecinos = np.arange(1,k)
    i_rep = i
    
    resultados_test_255 = np.zeros((i_rep,len(n_columnas), len(valores_k)))
    
    j = -1  # lo voy a usar para asignar en la posicion j de la matriz
    for n in n_columnas:
        j += 1
        col_mas_255 = n_col_mas_255(img_0_1,n)
        PIXELES_255 = col_mas_255.iloc[:,1:]
        DIGITO_255 = col_mas_255[0]
        for i in range(i_rep):
            X_train_255, X_test_255, Y_train_255, Y_test_255 = train_test_split(PIXELES_255,DIGITO_255, test_size = 0.3)
            for k in k_vecinos:
                model_255 = KNeighborsClassifier(n_neighbors = k)
                model_255.fit(X_train_255, Y_train_255)
                predicciones_test_255 = model_255.predict(X_test_255)
                precision_test_255 = metrics.accuracy_score(Y_test_255, predicciones_test_255)
                resultados_test_255[i,j,k-1] = precision_test_255

    promedio_precisiones = np.mean(resultados_test_255,axis=0)

    mayor_precision = np.max(promedio_precisiones)
    print('\nMayor precision alcanzada: ',mayor_precision)
    print()
    columnas_mayor_precision,vecinos_mayor_precision = np.unravel_index(np.argmax(promedio_precisiones), promedio_precisiones.shape)
    print('Mejor combinacion entre cantidad de columnas y vecinos:\n')
    print('Cantidad de columnas: ',n_columnas[columnas_mayor_precision],'\nCantidad de vecinos: ',vecinos_mayor_precision+1)

    # hago modelo con esos parametros para ver la matriz de confusion
    col_mas_255 = n_col_mas_255(img_0_1,n_columnas[columnas_mayor_precision])
    PIXELES_255 = col_mas_255.iloc[:,1:]
    DIGITO_255 = col_mas_255[0]
    X_train_255, X_test_255, Y_train_255, Y_test_255 = train_test_split(PIXELES_255,DIGITO_255, test_size = 0.3)
    model_255 = KNeighborsClassifier(n_neighbors = vecinos_mayor_precision+1)
    model_255.fit(X_train_255, Y_train_255)
    predicciones_test_255 = model_255.predict(X_test_255)
    print('\nMatriz de confusion para esos parametros:\n',metrics.confusion_matrix(Y_test_255,predicciones_test_255))
    print('\nPrecision: ',metrics.accuracy_score(Y_test_255,predicciones_test_255),'\n')


    sns.heatmap(promedio_precisiones, xticklabels=k_vecinos, yticklabels=n_columnas)
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Cantidad de columnas')
    plt.title('Precision del modelo para diferentes combinaciones de columnas y vecinos')
    plt.show()

#    return resultados_test_255

#-----------------------------------------------------------
# funcion de modelo KNN usando n columnas con mayor cantidad de pixeles 255

def knn_255(df,n_columnas,k_vecinos):
    df_255 = n_col_mas_255(df,n_columnas)
    PIXELES = df_255.iloc[:,1:]
    DIGITO = df_255[0]
    PIXELES_train, PIXELES_test, DIGITO_train, DIGITO_test = train_test_split(PIXELES,DIGITO, test_size = 0.3)
    model = KNeighborsClassifier(n_neighbors = k_vecinos)
    model.fit(PIXELES_train, DIGITO_train)
    PREDICCIONES = model.predict(PIXELES_test)
    print('----------------\nPrecision: \n')
    print(metrics.accuracy_score(DIGITO_test, PREDICCIONES))
    print('----------------\nMatriz de confusion\n')
    print(metrics.confusion_matrix(DIGITO_test, PREDICCIONES))
    print('----------------\n')




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




#plt.figure(figsize= [20,10])
#tree.plot_tree(clf_info, feature_names = ['altura_tot', 'diametro', 'inclinacio'], class_names = ['Ceibo', 'Eucalipto', 'Jacarandá', 'Pindó'],filled = True, rounded = True, fontsize = 8)
#
#plt.figure(figsize= [15,10])
#tree.plot_tree(clf_info, feature_names = iris['feature_names'], class_names = iris['target_names'],filled = True, rounded = True, fontsize = 10)
#
#r = tree.export_text(clf_info, feature_names=iris['feature_names'])
#print(r)
#
#
#clf_info = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= 4)
#clf_info = clf_info.fit(X, Y)
#
#
##%%
#
#datonuevo= pd.DataFrame([dict(zip(['altura_tot', 'diametro', 'inclinacio'], [22,56,8]))])
#clf_info.predict(datonuevo)
#
#
##%%
## otra forma de ver el arbol
#r = tree.export_text(clf_info, feature_names=['altura_tot', 'diametro', 'inclinacio'])
#print(r)
#
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




#--------------------------------------------------------------------------------

