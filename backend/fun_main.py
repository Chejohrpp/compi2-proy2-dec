import os

import pandas as pd
import numpy as np

def getName(file):
    root, extension = file.split('.')
    # print(extension)
    if(extension == 'csv'):
        return 'file.csv'
    elif(extension == 'json'):
        return 'file.json'
    elif(extension == 'xlsx'):
        return 'file.xlsx'
    elif(extension == 'xls'):
        return 'file.xls'
    else:
        return 'file'

def getDataFrame(file):
    if file == '':
        return pd.DataFrame()
    root, extension = os.path.splitext(file)
    if(extension == '.csv'):
        df = pd.read_csv(file)
        return df
    elif(extension == '.json'):
        df = pd.read_json(file)
        return df
    elif(extension == '.xlsx' or extension == '.xls'):
        df = pd.read_excel(file)
        return df
    else:
        return pd.DataFrame()


def getColumnas(file):
    df = getDataFrame(file)
    columnas = np.asarray(df.columns).tolist()
    return columnas


def getlistadoAnalysis():
    lista = {
        'Tendencia de la infección por Covid-19 en un Pais':{
            'caso':1,
            'name':'Tendencia de la infección por Covid-19 en un Pais',
            'no_parametros': 4,
            'parametros':['tiempo','confirmados','celda_pais'],
            'opcionales': ['celda_pais','nombre_pais'],
            'parametros_texto':['nombre_pais']
        },
        'Predicción de infectados en un Pais':{
            'caso':2,
            'name':'Predicción de infectados en un Pais',
            'no_parametros': 5,
            'parametros':['tiempo','confirmados','celda_pais'],
            'opcionales': ['celda_pais','nombre_pais'],
            'parametros_numericos':['tiempo_predecir'],
            'parametros_texto':['nombre_pais']
        },
        'Indice de Progresion de la pandemia':{
            'caso':3,
            'name':'Indice de Progresion de la pandemia',
            'no_parametros': 2,
            'parametros':['tiempo','confirmados']
        },
        'Prediccion de mortalidad por COVID en un Departamento':{
            'caso':4,
            'name':'Prediccion de mortalidad por COVID en un Departamento',
            'no_parametros': 5,
            'parametros':['tiempo','fallecido','celda_departamento'],
            'parametros_numericos':['tiempo_predecir'],
            'parametros_texto':['nombre_departamento']
        }
    }
    return lista

def listaAnalisisNombres():
    lista = ['Tendencia de la infección por Covid-19 en un Pais','Predicción de infectados en un Pais',
                'Indice de Progresion de la pandemia','Prediccion de mortalidad por COVID en un Departamento']
    return lista