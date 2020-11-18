# app.py
from flask import Flask, render_template, request
import cv2 #se importa la libreria para reconocimeinto de imagenes
import numpy as np #para los arreglos
import os #para utilizar los directorios
import matplotlib.pyplot as plt 

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import tensorflow.keras.optimizers as Optimizer




app = Flask(__name__)


@app.route('/')
def main():
    return render_template("app.html")


@app.route("/calculate", methods=['POST'])
def calculate():
    archivo=request.form['archivo']
    # p=archivo

    pandas_folder_path="/home/julian1026/Escritorio/img/tigres" #ruta de la carpeta 
    tigres=[] #lista que almacena las imagenes
    img_size=150 #tamaño de imagen
    for img in os.listdir(pandas_folder_path): 
        img = cv2.imread(os.path.join(pandas_folder_path,img))#lee las imagenes
        img_resize= cv2.resize(img,(img_size,img_size))# reduccion de las imagenes al tamaño
        tigres.append(img_resize) #a la lista le almacena las imagenes reducidas

    tigres = np.array(tigres)# la lista se convierte a un arreglo


    pandas_folder_path="/home/julian1026/Escritorio/img/pandas"
    pandast=[]
    img_size=150
    for img in os.listdir(pandas_folder_path):
        img = cv2.imread(os.path.join(pandas_folder_path,img))
        img_resize= cv2.resize(img,(img_size,img_size))
        pandast.append(img_resize)

    pandast = np.array(pandast)

    imagenes=np.concatenate([tigres,pandast])  #se concatena las dos carpetas
    Imagen = np.array(imagenes) #las convierte a un arreglo

    etiquetas_tigres = np.repeat(0,6) #se le da una etiqueta a las imagenes
    etiquetas_pandas = np.repeat(1,15)

    class_name=['Tigre','Panda']#mencionar las clases es decir la posicion en que se encuentra

    labels=np.concatenate([etiquetas_tigres,etiquetas_pandas])#se concatena las etiquetas
    Labels= np.array(labels)# los convierte a un arreglo

    #////////////////////////////////

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(150, 150,3)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(2, activation='softmax'),
    ])

    model.compile(optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']) #mide el porcentaje de casos que el modelo ha acertado
    model.fit(Imagen, Labels, epochs=10)#cantidad de iteracciones
    trained=model.fit(Imagen, Labels, epochs=10)
 

    img=cv2.imread("/home/julian1026/Documentos/imgPrueba/"+archivo)
    img_cvt=cv2.cvtColor(img, cv2.COLOR_BGR2RGB )
    plt.imshow(img_cvt)

    img2=img_cvt
    img2=cv2.resize(img2,(img_size,img_size))
    img2 = (np.expand_dims(img2,0))

    predictions_single =model.predict(img2)
    valor=class_name[np.argmax(predictions_single)]
    print(valor)


    return render_template('app.html', valor=valor)