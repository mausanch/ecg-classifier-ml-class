from distutils.log import debug
from flask import abort, Flask, jsonify, request
from flask import render_template, render_template_string, redirect
import logging.config
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler

import numpy
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from flask_cors import CORS, cross_origin

from datetime import datetime,timedelta

#today = datetime.now()
#print(today)


#fecha = datetime.strptime("2022-05-06", "%Y-%m-%d ")
#print (fecha)

scaler = MinMaxScaler()

# CONFIGURACIÓN DE VARIABLE PARA LA CREACIÓN DE SERVICIOS REST
service = Flask(__name__)
cors = CORS(service, resources={r"/*": {"origins": "*"}})

service.config['CORS_HEADERS'] = 'Content-Type'


# CONFIGURACIÓN DE LOG PARA PINTAR EL FLUJO
# logging.config.dictConfig(LOGGING)
log = logging.getLogger("mswitch")
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s %(name)-1s %(filename)-20s %(lineno)-4d %(levelname)-8s %(message)s')

fileHandler = ConcurrentRotatingFileHandler(
    'logs', maxBytes=1000000,
    backupCount=10)


handler.setFormatter(formatter)
log.addHandler(handler)
log.addHandler(fileHandler)

log.setLevel(logging.DEBUG)

data= pd.read_csv(r"dataset_csv.csv")

X = numpy.array(data[['age', 'Sex', 'Heart rate']])
y = numpy.array(data['Class code'])

scaler.fit_transform(X)

mt = joblib.load(r"filename.cls")

# METODO QUE LLAMA AL SCRIPT
@service.route('/', methods=['POST'])
@cross_origin()
def procesamiento_ecg():
    log.info('Procesamiento Iniciado')
    log.info("JSON RECIBIDO")
    log.info(request.form)
    log.info(request.headers)

    request_data = request.get_json()


    #fecha_nacimiento =request_data['edad']
    #fecha_nacimiento = datetime.strptime(fecha_nacimiento, "%Y-%m-%d ")

    #edad = timedelta(fecha_nacimiento) - today

    sexo = request_data['sexo']
    heart_rate = request_data['ritmo_cardiaco']
    edad = request_data['edad']

    x = [edad,sexo,heart_rate]
    arr = numpy.array(x)
    arr = arr[:,numpy.newaxis]
    arr = arr.reshape(1,-1)
    arr = scaler.transform(arr)
    print(arr)
    
    return str(mt.predict(arr))


if __name__ == '__main__':
    service.run(host="0.0.0.0", port=8090, debug=True)
