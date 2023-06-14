#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Título: trabajo práctico 2

Materia: Laboratorio de Datos

Grupo: NJL
"""

#%% Importacion de los modulos
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import tree
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt


#%% Carga de datos
df_img = pd.read_csv('mnist_desarrollo.csv',header=None)
df_img.columns = range(df_img.shape[1])
#%%
#==================================================================================
# EJERCICIO 1
#==================================================================================

print('\n====================================================\nEJERCICIO 1\n====================================================\n')

# 1. Realizar un anÃ¡lisis exploratorio de los datos. Ver, entre otras cosas,
# cantidad de datos, cantidad y tipos de atributos, cantidad de clases de la
# variable de interÃ©s (el dÃ­gito) y otras caracterÃ­sticas que consideren
# relevantes. Â¿CuÃ¡les parecen ser atributos relevantes? Â¿CuÃ¡les no? Se
# pueden hacer grÃ¡ficos para abordar estas preguntas.

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

# construyo un DataFrame para cada valor de numero representado en imagenes
# como no hay misma cantidad de imagenes (o muestras) para cada valor representado en imagenes
# puedo duplicar datos o recortar.
# decido recortar y tomar para cada imagen unicamente 5420 muestras que corresponde con la cantidad de imagenes que tiene el valor con menor cantidad de muestras
 
def dfss(df_img):
    dfs = []
    for i in range(len(np.array(df_img.iloc[:,0].value_counts()))):
        filtro_i = (df_img[0]==i)
        df_i = df_img[filtro_i]
        # recorto: 
        df_i = df_i.iloc[0:5420,:]
        dfs.append(df_i)
    return dfs

#dfs = dfss(df_img)
    

#Me fijo cuales atributos son los mas representativos: 
    
# creo una matriz para cada una de las columnas de los data frames (menos la primera que corresponde al número)

def matrices(dfs):
    mats = [] # cada matriz de aca adentro corresponde a una matriz con todas las columnas de una determinada posicion para cada numero, ponele todas las primeras columnas del df del 0, del 1, del 2 ...  
    for j in range(784): 
        L = []
        for i in range(len(dfs)):
            L.append([(dfs[i]).iloc[:,j+1]])
        mats.append(L)
    return mats
            
#mats = matrices(dfs) 
    
def distanciaTotales(mats):
    dist = []
    for i in range(len(mats)):
        disti = np.linalg.norm(mats[i])
        dist.append(disti)
    return dist
        
#distTotales = distanciaTotales(mats)
#columna = list(range(1,785))
#distTotales = pd.DataFrame(list(zip(columna,distTotales)), columns = ['columna','distTotales'])
#distTotales = distTotales.sort_values('distTotales',ascending=False)

# segun este data frame 'dist01' vemos qué columnas son las más representativas

#tres_distTotales = distTotales.head(3)

#%%
#==================================================================================
# EJERCICIO 2
#==================================================================================

print('\n====================================================\nEJERCICIO 2\n====================================================\n')

# 2. Construir un dataframe con el subconjunto que contiene solamente los
# dÃ­gitos 0 y 1.

#--------------------------------------------------------------------------------

img_0_1 = df_img[(df_img[0] == 0) | (df_img[0] == 1)]

#--------------------------------------------------------------------------------

#%%
#==================================================================================
# EJERCICIO 3
#==================================================================================

print('\n====================================================\nEJERCICIO 3\n====================================================\n')

# 3. Para este subconjunto de datos, ver cuÃ¡ntas muestras se tienen y
# determinar si estÃ¡ balanceado entre las clases.

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

# comparo la distancia euclidiana entre todas las columnas i del 0 y del 1 

from scipy.spatial import distance



def distancia(df_0, df_1):
    dist = []
    columna = []
    for i in range(784):
        disti = distance.euclidean(df_0.iloc[:,i+1], df_1.iloc[:,i+1])
        dist.append(disti)
        columna.append(i)
    return dist, columna
        


# segun este data frame 'dist01' vemos qué columnas son las más representativas

def n_col_mas_dist(img, n):
    df = dfss(img)
    df_0 = df[0]
    df_1 = df[1]
    dist, columna = distancia(df_0,df_1)
    dist01 = pd.concat([pd.Series(columna),  pd.Series(dist)], axis = 1)
    dist01 = dist01.sort_values(1,ascending=False)
    col_mas_distancia = img[np.insert(np.array(dist01.iloc[:n,0]),0,0)]
    return col_mas_distancia





#%%
graficos = []

def plot_roc_curve(graficos):
    for j in range(len(graficos)):
        fpr, tpr, _ = metrics.roc_curve(graficos[j][0],  graficos[j][1])
        plt.plot(fpr, tpr, label=graficos[j][2])
        plt.plot([0, 1], [0, 1], color='yellow', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc = 'best')
    plt.show()
    
    
n=3

col_al_azar = n_col_al_azar(img_0_1,n)
col_menos_ceros = n_col_menos_ceros(img_0_1,n)
col_equi_dist = n_col_equi_dist(img_0_1,n)
col_mas_253 = n_col_mas_253(img_0_1,n)
col_mas_255 = n_col_mas_255(img_0_1,n)
col_mas_var = n_col_mas_var(img_0_1,n)
col_mas_norma2 = n_col_mas_norma2(img_0_1,n)
col_mas_distancia = n_col_mas_dist(img_0_1,n)

k = 5

print('----------\nNumero de columnas: ',n,'\nNumero de vecinos: ',k,'\n-----------\n')

#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de menos ceros\n=========================')

PIXELES = col_menos_ceros.iloc[:,1:]
DIGITO = col_menos_ceros[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('menos ceros')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('mas 253')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('mas 255')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('al azar')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('equidistantes')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('mas varianza')))
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
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('mas norma dos')))
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')


#--------------------------------------------------------------------------------

print('=========================\nEntrenamiento KNN con columnas de mas distancia\n=========================')

PIXELES = col_mas_distancia.iloc[:,1:]
DIGITO = col_mas_distancia[0]

model = KNeighborsClassifier(n_neighbors = k) # modelo en abstracto
model.fit(PIXELES, DIGITO) # entreno el modelo con los datos PIXELES y DIGITO
PREDICCIONES = model.predict(PIXELES) # me fijo quÃ© clases les asigna el modelo a mis datos
graficos.append((DIGITO, PREDICCIONES, ('mas distancia')))
print('----------------\nPrecision: \n')
print(metrics.accuracy_score(DIGITO, PREDICCIONES))
print('----------------\nMatriz de confusion\n')
print(metrics.confusion_matrix(DIGITO, PREDICCIONES))
print('----------------\n')

#plot_roc_curve(graficos)
#--------------------------------------------------------------------------------

#%%
#==================================================================================
# EJERCICIO 5
#==================================================================================

print('\n====================================================\nEJERCICIO 5\n====================================================\n')

# 5. Para comparar modelos, utilizar validaciÃ³n cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el anÃ¡lisis
# de los resultados, tener en cuenta las medidas de evaluaciÃ³n (por ejemplo,
# la exactitud) y la cantidad de atributos.

#--------------------------------------------------------------------------------
# funcion para comparar criterios de seleccion de atributos
#cant_columnas: cantidad de columnas fijas
#tuplas_funciones : lista de abajo
# k_vecinos: tope de cant de vecinos
#k_folds cantidad de foldeos

def mejor_seleccion_columnas(imagenes,cant_columnas,tuplas_funciones,k_vecinos,k_folds):

    valores_k = range(1,k_vecinos)
    kf = StratifiedKFold(n_splits=k_folds,shuffle=True) 

    resultados_por_foldeo = []
    for train, test in kf.split(imagenes.iloc[:,1:],imagenes[0]):
        resultados_por_funcion = []
        for funcion in tuplas_funciones:
            # resultados test tendra k_folds elementos, donde cada uno es una lista de precisiones, una para cada cantidad de vecinos
            precisiones_por_cant_columna = []
            for n in range(len(cant_columnas)):
                indices = funcion[1](imagenes,cant_columnas[n]).columns[1:]                   
                PIXELES_train = imagenes.iloc[train,indices]
                DIGITO_train = imagenes.iloc[train,0]
                PIXELES_test = imagenes.iloc[test,indices]
                DIGITO_test = imagenes.iloc[test,0]
                precisiones_n_columnas = []
                for k in valores_k:
                    preciciones_vecinos = []
                    model = KNeighborsClassifier(n_neighbors = k)
                    model.fit(PIXELES_train, DIGITO_train)
                    PREDICCIONES_test = model.predict(PIXELES_test)
                    precision_test = metrics.accuracy_score(DIGITO_test, PREDICCIONES_test)
                    precisiones_n_columnas.append(precision_test) 
                precisiones_por_cant_columna.append(precisiones_n_columnas)
                
            resultados_por_funcion.append(np.mean(precisiones_por_cant_columna,axis=0))

        resultados_por_foldeo.append(resultados_por_funcion)
     
    resultados_promedio = np.mean(resultados_por_foldeo,axis=0)
    i=0
    for r in resultados_promedio: 
        plt.plot(valores_k, r, label = tuplas_funciones[i][0])
        i+=1

    plt.legend()
    plt.title('Exactitud del modelo de knn para diferentes seleccion de columnas, promediando preciciones entre 3, 5 y 8 columnas')
    plt.xlabel('Cantidad de vecinos')
    plt.ylabel('Exactitud (accuracy)')
    plt.show()


tuplas_todas_funciones = [('Columnas mas distancia',n_col_mas_dist),('Columnas  mas 255',n_col_mas_255),('Columnas mas varianza',n_col_mas_var),('Columnas mas 253',n_col_mas_253),('Columnas mas norma 2',n_col_mas_norma2),('Columnas al azar',n_col_al_azar),('Columnas equidistantes',n_col_equi_dist)]
tuplas_mejores_funciones = [('Columnas mas distancia',n_col_mas_dist),('Columnas  mas 255',n_col_mas_255),('Columnas mas varianza',n_col_mas_var)]

#graficos usados:
#mejor_seleccion_columnas(img_0_1,[3,5,8],tuplas_todas_funciones,10,3)
#mejor_seleccion_columnas(img_0_1,[3,5,8],tuplas_mejores_funciones,10,3)


# scatter para ver relacion entre cantidad de columnas y precision.
# cant_columnas: lista de cantidades de columnas
#funcion_columnas: n_col_...
#  k_vecinos = lista de cantidades de vecinos

def relacion_cant_columnas_precision(img,cant_columnas,funcion_columnas,k_vecinos,k_folds):
    kf = StratifiedKFold(n_splits=k_folds,shuffle=True) 
    resultados_por_foldeo = []
    indices = funcion_columnas(img,max(cant_columnas)).columns[1:]
    for train, test in kf.split(img.iloc[:,1],img[0]):
        resultados_por_vecino = []
        for k in k_vecinos:
            model = KNeighborsClassifier(n_neighbors = k)
        
            precisiones_por_columna = [] 
            for n in cant_columnas:

                PIXELES_train = img.iloc[train,indices[:n]]
                DIGITO_train = img.iloc[train,0]
                PIXELES_test = img.iloc[test,indices[:n]]
                DIGITO_test = img.iloc[test,0]
                
                model.fit(PIXELES_train, DIGITO_train)
                PREDICCIONES = model.predict(PIXELES_test)
                precision = metrics.accuracy_score(DIGITO_test, PREDICCIONES)
                precisiones_por_columna.append(precision)
                
            resultados_por_vecino.append(precisiones_por_columna)
        
        resultados_por_foldeo.append(resultados_por_vecino)
            
    promedio_resultados_foldeos = np.mean(resultados_por_foldeo, axis=0)

    for k in range(len(k_vecinos)):
        precisiones = promedio_resultados_foldeos[k]
        plt.scatter(cant_columnas,precisiones,label=str(k_vecinos[k])+' vecinos')


    plt.title('Relacion entre cantidad de columnas, vecinos y precision')
    plt.xlabel('Cantidad de columnas')
    plt.ylabel('Precision')
    plt.legend(loc='lower right')
    plt.show()

# graficos usados en el PDF:
#relacion_cant_columnas_precision(img_0_1,np.concatenate((np.arange(1,200,20),np.arange(220,780,80))),n_col_mas_dist,[3,12,25],3)
#relacion_cant_columnas_precision(img_0_1,np.concatenate((np.arange(1,200,20),np.arange(220,780,80))),n_col_mas_255,[3,12,25],3)
#relacion_cant_columnas_precision(img_0_1,np.concatenate((np.arange(1,200,20),np.arange(220,780,80))),n_col_mas_var,[3,12,25],3)



# mapa de calor relacion entre cantidad de vecinos y columnas
# funcion_columnas: n_col_... funcion para seleccionar columnas
# n_columnas: lista de cantidades de columnas
# k: tope de cantidad de vecinos
# k_folds cantidad de foldeos para mejorar precision de promedio
# nombre_funcion: string '' del nombre de la funcion para seleccionar columnas

def mapa_de_calor(img,funcion_columnas,n_columnas,k,k_folds, nombre_funcion):
    k_vecinos = np.arange(1,k)
    
    resultados_por_foldeo = []
    kf = StratifiedKFold(n_splits=k_folds,shuffle=True) 
    indices = funcion_columnas(img,max(n_columnas)).columns[1:]
    for train, test in kf.split(img.iloc[:,1],img[0]):
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

            resultados_por_columna.append(resultados_por_vecinos)

        resultados_por_foldeo.append(resultados_por_columna)
                    

    print(resultados_por_foldeo)
    promedio_precisiones = np.mean(resultados_por_foldeo,axis=0)
    print(promedio_precisiones)

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


# graficos usados en el PDF:
#mapa_de_calor(img_0_1,n_col_mas_255,np.arange(50,150,6),10,2,'255')
#mapa_de_calor(img_0_1,n_col_mas_var,np.arange(1,50,3),10,2,'255')
#mapa_de_calor(img_0_1,n_col_mas_dist,np.arange(1,50,3),10,2,'255')


# funcion para entrenar modelo de N columnas y K vecinos, y testearlo sobre mnist_test_binario

def knn_train(img,test,cant_columnas,k_vecinos,funcion_columnas):
     indices = funcion_columnas(img,cant_columnas).columns[1:]
     PIXELES_test = test.iloc[:,indices]
     DIGITO_test = test[0]
     PIXELES_train = img.iloc[:,indices]
     DIGITO_train = img[0]
     model = KNeighborsClassifier(n_neighbors = k_vecinos)
     model.fit(PIXELES_train, DIGITO_train)
     PREDICCIONES = model.predict(PIXELES_test)
     print('----------------\nPrecision: \n')
     print(metrics.accuracy_score(DIGITO_test, PREDICCIONES))
     print('----------------\nMatriz de confusion\n')
     print(metrics.confusion_matrix(DIGITO_test, PREDICCIONES))
     print('----------------\n')
     return metrics.accuracy_score(DIGITO_test, PREDICCIONES)


# testeos finales sobre test_binario usando ambos criterios
test = pd.read_csv('mnist_test_binario.csv',header=None)
test.columns = range(test.shape[1])

print('\nTesteo criterio mayor distancia, 35 pixeles y 7 vecinos\n')
knn_train(img_0_1,test,35,7,n_col_mas_dist)
print('\nTesteo criterio mas 255, 78 pixeles y 1 vecinos\n')
knn_train(img_0_1,test,78,1,n_col_mas_255)
print('\nTesteo criterio mas varianza, 35 pixeles y 2 vecinos\n')
knn_train(img_0_1,test,35,2,n_col_mas_255)


#%%
#==================================================================================
# EJERCICIO 6
#==================================================================================

print('\n====================================================\nEJERCICIO 6\n====================================================\n')

# 6. Trabajar nuevamente con el dataset de todos los dÃ­gitos. Ajustar un
# modelo de Ã¡rbol de decisiÃ³n. Analizar distintas profundidades.

#--------------------------------------------------------------------------------


def distintasProfundidades(mnistD,valores_n):
    X = mnistD.iloc[:,1:]
    Y= mnistD.iloc[:,0]
    Y = Y.to_frame()
    performance_accuracy_testT = []
    performance_accuracy_trainT = []
    #los indexes + 1 corresponden a la profundidad.

    for i in valores_n:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
        clf_info = tree.DecisionTreeClassifier(criterion = "entropy", max_depth= i)
        clf_info = clf_info.fit(X_train, Y_train)
        Y_pred = clf_info.predict(X_test)
        Y_pred_train = clf_info.predict(X_train)
        acc_test = metrics.accuracy_score(Y_test, Y_pred)
        acc_train = metrics.accuracy_score(Y_train, Y_pred_train)
        performance_accuracy_testT.append(acc_test)
        performance_accuracy_trainT.append(acc_train)
#Tomo en consideracion los resultados en test para hacer los graficos.
#Graficos para decidir mejor seleccion de profundidad

    plt.plot(valores_n, performance_accuracy_testT)
    plt.xlabel('Posibles Valores de profundidad')
    plt.ylabel('Accuracy')
    plt.title('Accuracy en función de los valores de profundidad')
    plt.show()
    plt.close()

'''
# printeo imagen 1
distintasProfundidades(df_img, range(1,51,5))

# printo iomagen 3
distintasProfundidades(df_img, range(10,16,1))

'''

#%%
#==================================================================================
# EJERCICIO 7
#==================================================================================

print('\n====================================================\nEJERCICIO 7\n====================================================\n')

# 7. Para comparar y seleccionar los arboles de decision, utilizar validación
# cruzada con k-folding.

#--------------------------------------------------------------------------------

def selectArbolACC(X, Y, profundidad, criterio):
    # armo una lista donde voy a guardar los scores para cada fold
    acc = []
    # defino el número de folds
    k = 5
    # creo el objeto KFold
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train, test in kf.split(X, Y):
        # lo que hace es hacerme un split de los datos en 10 folds.
        # en la variable train los indices de las filas que corresponden a los datos TRAIN
        # en la variable test los indices de las filas que corresponden a los datos TEST
        X_train =   X.iloc[train]
        Y_train = Y.iloc[train]
        X_test = X.iloc[test]
        Y_test = Y.iloc[test]
        # creo los DF que voy a usar para entrenar y los que voy a usar para testear
        c = []
        for j in range(len(criterio)):
            accK = [0]
            # para cada split voy a guardad una lista donde cada elemento de esa lista me da el score de clasificar con una profundidad determinada
            # arranco con un cero porque quiero que la columna se corresponda con la profundidad elegida y nunca selecciono profundidad = 0
            for i in range(0, profundidad, 4):
                num_info = tree.DecisionTreeClassifier(criterion = criterio[j], max_depth=i+1) # calcula la profundidad desde el 1 al profundidad
                num_info = num_info.fit(X_train, Y_train)
                Y_pred_test = num_info.predict(X_test)
                accuracy = metrics.accuracy_score(Y_pred_test , Y_test)
                accK.append(accuracy)
            c.append(accK)
        acc.append(c)
    return acc

'''
# printo imagen 2
X = df_img.iloc[:, 1:] # los atributos
Y = df_img.iloc[:, 0] # los numeros representados por los atributos X Pacc = pd.DataFrame(Pacc, columns = criterio)

profundidad = 30
criterio = ['entropy', 'gini']

acc = selectArbolACC(X, Y, profundidad, criterio)

acc2= np.array(acc)
acc_p= np.mean(acc2,axis=0).T

# grafico:

Pacc = pd.DataFrame(acc_p,columns = criterio)

cortes = np.arange(0,profundidad, 4)

plt.plot(cortes, Pacc.loc[1:,'gini'], color = 'green', label = 'gini')
plt.plot(cortes, Pacc.loc[1:,'entropy'], color = 'pink', label = 'entropy')
plt.xlabel('Posibles Valores de profundidad')
plt.ylabel('Accuracy')
plt.title('Accuracy con criterio Gini y Entropy distintas profundidades')
plt.legend(loc = 'best')
'''


#%%

# ARMADO DE MODELOS  FINALES

X = df_img.iloc[:, 1:] # los atributos
Y = df_img.iloc[:, 0] # los dígitos

X_test = test.iloc[:, 1:]
Y_test = test.iloc[:, 0]

best_tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 15)
best_tree = best_tree.fit(X, Y)
Y_pred_test = best_tree.predict(X_test)
best_accuracy_arbol = metrics.accuracy_score(Y_pred_test , Y_test)

print('\nPrecision de modelo de arbol para criterio entropia y profundidad 15: \n',best_accuracy_arbol,'\n')

