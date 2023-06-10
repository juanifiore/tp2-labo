'''
Trabajo practico Nº 2, Laboratorio de Datos
Grupo NJL: Falczuk Noelia, Sanes Salazar Luna, Fiore Juan Ignacio

El codigo implementara funciones de la libreria SKLearn para intentar lograr una clasificacion
de imagenes de numeros escritos a mano. Utilizando el csv mnist_desarrollo.csv que contiene un 
conjunto de 

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


#%%

df_img = pd.read_csv('mnist_desarrollo.csv',header=None)
df_img.columns = range(df_img.shape[1])

#%%
#================================================================================
# EJERCICIO 1
#================================================================================

print('\n=======================================\nEJERCICIO 1\n=======================================\n')

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
sns.barplot(x='numero',y='cantidad',data=pd.DataFrame({'numero': cantidad_de_imagenes_por_numero.index, 'cantidad': cantidad_de_imagenes_por_numero.values}))
plt.axhline(cantidad_de_imagenes_por_numero.mean(), color='red', linestyle='--')
plt.show()
plt.close()


#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------

#--------------------------------------------------------------------------------


#%%
#================================================================================
# EJERCICIO 2
#================================================================================

print('\n=======================================\nEJERCICIO 2\n=======================================\n')

# 2. Construir un dataframe con el subconjunto que contiene solamente los
# dígitos 0 y 1.

#--------------------------------------------------------------------------------

img_0_1 = df_img[(df_img[0] == 0) | (df_img[0] == 1)]

#--------------------------------------------------------------------------------

#%%
#================================================================================
# EJERCICIO 3
#================================================================================

print('\n=======================================\nEJERCICIO 3\n=======================================\n')

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


#--------------------------------------------------------------------------------


#%%
#================================================================================
# EJERCICIO 4
#================================================================================

print('\n=======================================\nEJERCICIO 4\n=======================================\n')

# 4. Ajustar un modelo de knn considerando pocos atributos, por ejemplo 3.
# Probar con distintos conjuntos de 3 atributos y comparar resultados.
# Analizar utilizando otras cantidades de atributos.

#--------------------------------------------------------------------------------

# armaremos nuevos dataframe que contengan solo n atributos (pixeles o columnas)
# primero n al azar, luego n equidistantes, y luego las n columnas que contengan mayor cantidad de datos distintos de cero

# al azar
def n_col_al_azar(img,n):
    indices_al_azar = np.random.randint(1, 785, size=n)
    col_al_azar = img_0_1.iloc[:,np.insert(indices_al_azar,0,0)]
    return col_al_azar

# menos cantidad de ceros
def n_col_menos_ceros(img,n):
    cantidades_de_ceros = (img_0_1.iloc[:,1:]==0).sum()
    indices_n_columnas_menos_ceros = cantidades_de_ceros.sort_values().index[:n]
    col_menos_ceros = img_0_1[np.insert(indices_n_columnas_menos_ceros,0,0)]    # le agrego la columna 0
    return col_menos_ceros

# equidistantes (n columnas que parten los array en (n+1) partes iguales)
def n_col_equi_dist(img,n):
    indices_equi_dist = np.arange(784/(n+1), 784, np.ceil(784/(n+1)),dtype=int)
    col_equi_dist = img_0_1[np.insert(indices_equi_dist,0,0)]   # le agrego la columna 0 
    return col_equi_dist

n=10

col_al_azar = n_col_al_azar(img_0_1,n)
col_menos_ceros = n_col_menos_ceros(img_0_1,n)
col_equi_dist = n_col_equi_dist(img_0_1,n)


#--------------------------------------------------------------------------------

k = 8

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
#================================================================================
# EJERCICIO 5
#================================================================================

print('\n=======================================\nEJERCICIO 5\n=======================================\n')

# 5. Para comparar modelos, utilizar validación cruzada. Comparar modelos
# con distintos atributos y con distintos valores de k (vecinos). Para el análisis
# de los resultados, tener en cuenta las medidas de evaluación (por ejemplo,
# la exactitud) y la cantidad de atributos.

#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------


#%%
#================================================================================
# EJERCICIO 6
#================================================================================

print('\n=======================================\nEJERCICIO 6\n=======================================\n')

# 6. Trabajar nuevamente con el dataset de todos los dígitos. Ajustar un
# modelo de árbol de decisión. Analizar distintas profundidades.

#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------


#%%
#================================================================================
# EJERCICIO 7
#================================================================================

print('\n=======================================\nEJERCICIO 7\n=======================================\n')

# 7. Para comparar y seleccionar los árboles de decisión, utilizar validación
# cruzada con k-folding.

#--------------------------------------------------------------------------------



#--------------------------------------------------------------------------------

