import os
import sys
from flask import Flask, jsonify,request,url_for
from flask_cors import CORS, cross_origin
from werkzeug.utils import redirect
# from reportes.Prediccion import impirmir
import reportes.funciones.Primarias as pr
import fun_main as fm
import Analisis as an

ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
# print ('root paht',ROOT_PATH)

os.environ.update({'ROOT_PATH': ROOT_PATH})
os.environ.update({'ENV': "desarrollo"})
os.environ.update({'PUERTO': '4000'})
sys.path.append(os.path.join(ROOT_PATH,'reportes'))

app = Flask(__name__)
CORS(app)

app.config["file_analizar"] = 'static/files_analizar'
app.config["path_file"] = 'static/files_analizar/file.xlsx' ##Esto tiene que quedar nulo
app.config["file_name"] = 'file.xlsx'  ##Esto tiene que quedar nulo
app.config["datos_reporte"] = {}

@app.route("/",methods=['GET'])
def inicio():
    return jsonify(configReturn('Bienvenidos al api'))

@app.route("/api",methods=['GET'])
def hello_world():
    # return jsonify({"hello":"hello world","atributo":224})
    # print(app.config["path_file"])
    return jsonify(configReturn(''))


@app.route("/api",methods=['POST'])
def recibirdata():
    
    if request.files:
        data = request.files["file_data"]
        name = fm.getName(data.filename)
        app.config["file_name"] = data.filename
        data.save(os.path.join(app.config["file_analizar"],name))
        # print("data saved")
        app.config["path_file"]  = app.config["file_analizar"] + '/' + name
        # print(app.config["path_file"])
        columnas = fm.getColumnas(app.config["path_file"])
        columnas_json = {"columnas": columnas}
        return jsonify(configReturn(columnas_json))

    return jsonify(configReturn(''))

@app.route('/api/parametros',methods=['GET'])
def enviar_parametros():
    listaAnalisis = fm.getlistadoAnalysis()
    file_name =  app.config["file_name"]
    columnas = fm.getColumnas(app.config["path_file"])
    listaNombre =  fm.listaAnalisisNombres()
    body = {"listaAnalisis": listaAnalisis,"fileName":file_name,"columnas":columnas,'listaNombres':listaNombre}
    return jsonify(configReturn(body))

@app.route('/api/parametros',methods=['POST'])
def realizarAnalisis():
    if app.config["path_file"] != '':
        # print(request.json)
        # print(request.json['caso'])
        # print(request.json["name"])
        # print(request.json["parametros"])
        datos_reporte = an.redirigirAnalisis(app.config["path_file"],request.json["caso"],request.json["name"],request.json["parametros"])
        app.config["datos_reporte"] = datos_reporte
    return jsonify(configReturn(''))

@app.route('/api/reporte',methods=['POST','GET'])
def enviar_reporte():
    #realizar los envios de imgs
    if len(app.config["datos_reporte"]) != 0:
        return configReturn(app.config["datos_reporte"])    
    return configReturnStatus(204,'')

@app.route('/api/static/imgs_temp/<name_img>',methods=['GET'])
def redirigirImgs(name_img):
    #'/static/imgs_temp/'+name_img
    return redirect('/static/imgs_temp/'+name_img)

def configReturn(body):
    retorno = {
        'status':200,
        'headers':{
            'Access-Control-Allow-Origin': '*',
        },
        'body': body        
    }
    return retorno

def configReturnStatus(status,body):
    retorno = {
        'status':status,
        'headers':{
            'Access-Control-Allow-Origin': '*',
        },
        'body': body        
    }
    return retorno

### Esto deberia ser cambiado ###
app.config['DEBUG'] = os.environ.get('ENV') == 'desarrollo'
app.run(host='localhost',port=int(os.environ.get('PUERTO')))