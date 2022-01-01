import os

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import fun_main as fm

path_imgs = 'static/imgs_temp'

def analizar(filepath,x_celda,y_celda,pais_celda=None,pais=None):
    lista_urls_imgs = []
    lista_urls_static = []
    datos_calculados = []
    datos_estaticos = []
    ### GET DataFrame  ###############################################
    df = fm.getDataFrame(filepath)
    if(df.empty):
        print ('Error, no hay un dataframe')
        return False

    ######### Limpiar los datos ##########################################
    # df[y_celda] = pd.to_numeric(df[y_celda].str.split(',').str.join(''))

    if pais != "" and pais_celda != "":
        df = df[df[pais_celda].str.contains(pais)] ## country/region  se cambia
    
    ##### Asginamos Variables ##########################################
    x = np.asarray(df[x_celda]).reshape(-1,1)
    y = df[y_celda]

    # ##datos extras##
    plt.scatter(df[x_celda],df[y_celda],color="blue")
    path_aux = generarUrlImg("fig_muestra.png",lista_urls_static)
    plt.savefig(path_aux)
    plt.clf()

    #### build ###############################################################
    grado = 2
    poly_feature = PolynomialFeatures(grado)
    x_transform = poly_feature.fit_transform(x)

    #### Train ###############################################################
    #algorithm
    l_reg = linear_model.LinearRegression()

    model = l_reg.fit(x_transform,y)
    y_predictions = model.predict(x_transform)

    #### Calculate ###########################################################
    rmse = np.sqrt(mean_squared_error(y,y_predictions))
    print("rmse:",rmse)
    datos_calculados.append("rmse : " + str(rmse))
    datos_estaticos.append("rmse : " + str(rmse))
    r2 = r2_score(y,y_predictions)
    print ("r^2:",r2)
    datos_calculados.append("r^2 : " + str(r2))
    datos_estaticos.append("r^2 : " + str(r2))

    #### Prediccion ##########################################################
    ## no se realiza predicion aqui

    #### Graph #######################################################################
    title = 'grado usado {}; RMSE = {}; R^2={:.3f}'.format(grado,round(rmse,2),r2)
    plt.title("Tendencia de la infección por Covid-19 en un Pais\n"+title,fontsize=10)
    plt.xlabel(x_celda)
    plt.ylabel(y_celda)
    plt.plot(df[x_celda],y_predictions,color="red",linewidth=3)
    path_aux = generarUrlImg("fig_tendencia.png",lista_urls_imgs)
    plt.savefig(path_aux)
    plt.clf()
    # plt.show()
    return addData(datos_calculados,lista_urls_imgs,lista_urls_static,datos_estaticos)

def generarUrlImg(name,lista_urls_imgs):
    url_temp = path_imgs + '/' + name
    lista_urls_imgs.append(url_temp)
    return url_temp

def addData(datos_calculados,lista_urls_imgs,lista_urls_static,datos_estaticos):
    lista_datos ={
        'lista_urls': lista_urls_imgs,
        'calculos_reporte': datos_calculados,
        'lista_urls_static':lista_urls_static,
        'conclusion': 'No hay conclusion',
        "datos_statics":datos_estaticos
    }
    return lista_datos


