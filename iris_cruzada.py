#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 17:05:49 2019

@author: viniciusmaurente
"""
#importação das bibliotecas
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import model_from_json
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

#carregamento e tratamento da base de dados
base = pd.read_csv('iris.csv')
previsores = base.iloc [:, 0:4].values
classe = base.iloc [:,4].values
labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
classe_dummy = np_utils.to_categorical(classe)

#criação da estrutura da rede neural e treinamento
def criar_rede():
    classificador = Sequential()
    classificador.add(Dense(units=8, activation = 'relu',
                            kernel_initializer='normal', input_dim = 4))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=8, activation = 'relu',
                            kernel_initializer='normal',))
    classificador.add(Dropout(0.1))
    classificador.add(Dense(units=3, activation = 'softmax'))
    classificador.compile(optimizer='adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificador 

classificador = KerasClassifier(build_fn = criar_rede,
                                epochs = 2000,
                                batch_size = 10)

resultado = cross_val_score(estimator = classificador, 
                            X = previsores, y = classe,
                            cv = 10, scoring = 'accuracy')
#Salvar o classificador
classificador_json = classificador.to_json()
with open ("classificador_iris.json", "w") as json_file:
            json_file.write(classificador_json)
classificador.save_weights("classificador_iris.h5")


media = resultado.mean()
desvio = resultado.std()



