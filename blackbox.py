# Imports
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

import os, sys
import base64
from numpy import array
import json
from socketIO_client import SocketIO
cd = os.path.dirname(os.path.abspath(__file__))
os.chdir(cd)

sys.path.insert(0, os.path.join(cd, 'libs'))

import random

# Server parameters
host = '10.1.196.58'
port = 5000


# {
#     'command': 'update_preferences', 
#     'preferences': '4'
# }


map_to_classes= {
    'Corner': 'corner', 
    'CounterAttack': 'oppscore', 
    'GoalAtack': 'wescore', 
    'Goalkeeper': 'oppscore', 
    'LogoView': 'corner', 
    'MidField': 'offside', 
    'OutField': 'corner', 
    'Penalty': 'penalty', 
    'Player(Close-up)': 'wescore',
    'Referee': 'penalty'
}


map_to_zooms = {
    'Corner': 'corner',
    'CounterAttack': 'steal', 
    'GoalAtack': 'goal', 
    'Goalkeeper': 'save', 
    'LogoView': 'foul', 
    'MidField': 'offside', 
    'OutField': 'assist', 
    'Penalty': 'penalty', 
    'Player(Close-up)': 'fight',
    'Referee': 'freek'
}

# class_to_out{
#     'Attacker': ,
#     'Defenders': ,
#     'Ball': ,
#     'Goalkeeper':
# }

userData = ['Male', 1,   1,   4,   4,   4,   1,   5,   5,   5,   "'Whole game'",    2,   3,   1,   1,   3,  
 'Together',    4,   2,   1,   3,   3,   2,   2,   4,   3,   2,   3,   1,   1,   2];

# Detection and tracking parameters
detection_frequency = 30  # Detects every 30 frames


class IntelligentProcessingBlackbox:
    def __init__(self, io):
        self.io = io
        self.models = {}
        self.models_zoom = {}

    def controller(self, data):
        print('--------------')
        print(data)
        # TODO parse info


        # add emotion to record
        userData.append(emotion)

        # map classes
        scene_mapped = map_to_classes[scene]
        zoom_scene_mapped = map_to_classes[zoom_scene]

        cl, zoom = self.predict(userData,scene_mapped,zoom_scene_mapped)

        userData.pop()

        #  TODO pack results
        res = {
            'obj': cl,
            'zoom': zoom
        }


        self.io.send(json.dumps(res))

    def columnTransform(self, data):
        transformer = ColumnTransformer(
            transformers=[
                ("OneHot",        # Just a name
                 OneHotEncoder(), # The transformer class
                 [0, 10, 16]              # The column(s) to be applied on.
                 )
            ]
        )
        X_categ = transformer.fit_transform(data)
        X = np.delete(X,[0,10,16],1)
        X = np.concatenate((X,X_categ),axis=1)
        return X

    def train():
        data = pd.read_csv('./data.csv')
        df = pd.DataFrame(data)

        np_data = df.values
        cols = list(df)
        self.feature_cols = cols[:-15]
        self.target_cols = cols[-15:]

        X = np_data[:, [i for i in range(len(feature_cols))]]
        X = self.columnTransform(X)

        Y_classes = {
            'offside' : np_data[:, -5],
            'penalty' : np_data[:, -4],
            'corner'  : np_data[:, -3],
            'oppscore': np_data[:, -2],
            'wescore' : np_data[:, -1]
        }
        Y_zoom = {
            'corner' : np_data[:, -15],
            'save' : np_data[:, -14],
            'freek'  : np_data[:, -13],
            'goal': np_data[:, -12],
            'assist' : np_data[:, -11],
            'foul' : np_data[:, -10],
            'penalty' : np_data[:, -9],
            'offside' : np_data[:, -8],
            'steal' : np_data[:, -7],
            'fight'  : np_data[:, -6]
        }

        for cl in Y_classes:
            knn = KNeighborsClassifier(n_neighbors=25)
            knn.fit(X, Y_classes[cl])
            self.models[cl] = knn

        for cl in Y_zoom:
            knn = KNeighborsRegressor(n_neighbors=25)
            knn.fit(X, Y_zoom[cl])
            self.models_zoom[cl] = knn

    def predict(self,records,scene,zoom_scene):
        records = self.columnTransform([records])
        cl = self.models[scene].predict(records[0])
        zoom = self.models[zoom_scene].predict(records[0])
        return cl, zoom

    def run(self):
        self.io.connect()
        print("    -----    socket connected")
        self.io.listen(self.controller)
        self.io.disconnect()



class SocketInputOutput:
    def __init__(self, host, port):

        self.host = host
        self.port = port
        print("fdsfdfs")
        self.socketio = SocketIO(host, port)
        print("top line is not working")

    def connect(self):
        self.socketio.on('connect', self.connected_callback)
        self.socketio.on('reconnect', self.reconnected_callback)
        self.socketio.on('disconnect', self.disconnected_callback)
        self.socketio.emit('join', 'IP')

    def disconnect(self):
        self.socketio.emit('leave', 'IP')

    def listen(self, x):
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        self.socketio.on('IP', x)
        print("wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww")
        self.socketio.wait()

    def send(self, data):
        self.socketio.emit('IP', data)

    def connected_callback(self):
        print("Connected to server: " + self.host + ":" + self.port + "!")

    def reconnected_callback(self):
        print("Reconnected to server: " + self.host + ":" + self.port + "!")

    def disconnected_callback(self):
        print("Disconnected from server: " + self.host + ":" + self.port + "!")
   
    def objectupdate(self, data):
        return objects

    def listener(self, data):
        print('received => ', len(data))
        objects=self.objectupdate(data)
        self.socketio.emit('IP', objects)
        print('sent => ', objects)


if __name__ == '__main__':
    print('sdgdsgdsg')
    print(__name__)
    io = SocketInputOutput(host, port)
    print("here")
    IntelligentProcessingBlackbox(io).run()

