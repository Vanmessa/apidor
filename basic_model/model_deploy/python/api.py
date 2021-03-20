from tensorflow.keras.models import load_model
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import sqlite3
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json


app = FastAPI()

class Data(BaseModel):
    UserID:str
    # x:float
    # y:float

def loadModel():
    global predict_model
    predict_model = load_model('model.h5')
    # predict_model = load_model('model1.h5')

loadModel()

def db_connect():
    con = sqlite3.connect('dor_rec.sqlite3')
    print("connect success")
    return con

async def predict(data):
    con=db_connect()
# con.row_factory = dict_factory
    cur= con.cursor()
    combined = pd.read_sql_query("SELECT * FROM combined", con)
# combined
    combined_df = pd.DataFrame(data = combined)
# combined_df
    combined_df['DorRating'] = combined_df['DorRating'].values.astype(float)
# combined_df
    combined_df = combined_df.drop_duplicates(['UserID', 'Dormitory'])
# combined_df
    user_dor_matrix = combined_df.pivot(index='UserID', columns='Dormitory', values='DorRating')
    user_dor_matrix.fillna(0, inplace=True)
# user_dor_matrix
    users = user_dor_matrix.index.tolist()
# users
    dors = user_dor_matrix.columns.tolist()
# dors
    user_dor_matrix = user_dor_matrix.to_numpy()
# user_dor_matrix
    filename = 'model.h5'
    predict_model = load_model(filename) 
# predict_model.summary()

    preds = predict_model(user_dor_matrix)
    preds = preds.numpy()
    pred_data = pd.DataFrame(preds)
# preds
    pred_data = pred_data.stack().reset_index()
# pred_data
    pred_data.columns = ['UserID', 'Dormitory', 'DorRating']
# pred_data
    pred_data['UserID'] = pred_data['UserID'].map(lambda value: users[value])
    pred_data['Dormitory'] = pred_data['Dormitory'].map(lambda value: dors[value])
# pred_data
# get_all_users
# get_all_users(json_str = True)
    cur.execute("SELECT UserID, Dormitory FROM pred_data ORDER BY UserID, DorRating ASC")
    data1=cur.fetchall()
# type(data1)
# data1
# dict1={}
# dict1=data1.to_dict('index')
# dict1
    d = {}
    for key, val in data1:
        d.setdefault(key, []).append(val)

    # print(d)
    # json_object = json.dumps(d, ensure_ascii=False)
    # n=data.UserID
    userid=data.UserID
    n = int(data.UserID)
    dor=d.get(n)

    return  userid,dor 

@app.post("/getclass/")
async def get_class(data: Data):
    # category, confidence = await predict(data)
    # res = {'class': category, 'confidence':confidence}

    userid,dor = await predict(data)
    res = { 'UserID':userid,'dormitory':dor}
    return {'results': res}

