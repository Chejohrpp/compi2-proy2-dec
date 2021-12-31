import os

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import fun_main as fm


def analizar(filepath,x_celda,y_celda,pais_celda=None,pais=None):
    ### GET DataFrame  ###############################################
    df = fm.getDataFrame(filepath)
    if(df.empty):
        print ('Error, no hay un dataframe')
        return False

    ######### Limpiar los datos ##########################################
    if pais != None:
        df = df[df[pais_celda].str.contains(pais)] ## country/region  se cambia
    
    ##### Asginamos Variables ##########################################
    x = np.asarray(df[x_celda]).reshape(-1,1)
    y = df[y_celda]

    #### build ###############################################################
    grado = 3
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
    r2 = r2_score(y,y_predictions)
    print ("r^2:",r2)

    #### Prediccion ##########################################################
    ## no se realiza predicion aqui

    #### Graph #######################################################################
    title = 'grado usado {}; RMSE = {}; R^2={:.3f}'.format(grado,round(rmse,2),r2)
    plt.title("Tendencia de la infección por Covid-19 en un Pais\n"+title,fontsize=10)
    plt.xlabel(x_celda)
    plt.ylabel(y_celda)
    plt.plot(df[x_celda],y_predictions,color="red",linewidth=3)
    # plt.savefig("fig1.png")
    plt.show()
