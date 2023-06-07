import os
import shutil
from flask import *
import json, time
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
tf.get_logger().setLevel('ERROR')


preprocessor_handle = "bert_en_uncased_preprocess_3"
model_handle = 'TF2'

transformer_encoder = hub.KerasLayer(model_handle, trainable=False)
preprocessor = hub.KerasLayer(preprocessor_handle)

# text = ["This is an example sentence."]
# preprocessed_text = preprocessor_handle(text)
# print(preprocessed_text)

class TextClassificationModel(tf.keras.Model):
    def __init__(self, PREPROCESS_MODEL, BERT_MODEL):
        super().__init__()
        self._preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
        self._transformer_encoder = hub.KerasLayer(BERT_MODEL, trainable=True)
        self._hidden_layer1 = tf.keras.layers.Dense(units=256, activation='relu')
        self._hidden_layer2 = tf.keras.layers.Dense(units=128, activation='relu')
        self._classifier = tf.keras.layers.Dense(units=8, activation='softmax')

    def call(self, inputs):
        preprocessed_inputs = self._preprocessor(inputs)
        embedding = self._transformer_encoder(preprocessed_inputs)
        hidden1_output = self._hidden_layer1(embedding)
        hidden2_output = self._hidden_layer2(hidden1_output)
        logits = self._classifier(hidden2_output)
        return logits


optimizer = tf.keras.optimizers.Adam(3e-5)
model = TextClassificationModel(preprocessor_handle, model_handle)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=[tf.metrics.CategoricalAccuracy()]
)

model = TextClassificationModel(preprocessor_handle, model_handle)

# Restore the weights
model.load_weights('Trained/Weights/Weights')

def call_model(text):
  results = model.call([text])
  res_array = np.argmax(results, axis=1)
  res = []
  {'joy': [1, 0, 0, 0, 0, 0, 0, 0], 'neutral': [0, 1, 0, 0, 0, 0, 0, 0], 'sadness': [0, 0, 1, 0, 0, 0, 0, 0],
   'anger': [0, 0, 0, 1, 0, 0, 0, 0], 'surprise': [0, 0, 0, 0, 1, 0, 0, 0], 'love': [0, 0, 0, 0, 0, 1, 0, 0],
   'fear': [0, 0, 0, 0, 0, 0, 1, 0], 'disgust': [0, 0, 0, 0, 0, 0, 0, 1]}
  for test_value in res_array:
    if (test_value == 0):
      res.append('joy')
    elif (test_value == 1):
      res.append('neutral')
    elif (test_value == 2):
      res.append('sadness')
    elif (test_value == 3):
      res.append('anger')
    elif (test_value == 4):
      res.append('suprise')
    elif (test_value == 5):
      res.append('love')
    elif (test_value == 6):
        res.append('fear')
    else:
      res.append('disgust')
  print(res)
  return res_array, res

app = Flask(__name__)

@app.route("/", methods = ['GET'])
def home_page():
    data_set = {'Page':'Home', 'Message':'Connected', 'Timestamp':time.time()}
    json_dump = json.dumps(data_set)

    return json_dump

@app.route("/emotion/", methods = ['POST'])
def check_emotions():
    user_query = str(request.args.get('user')) # /user/?user=UserName
    data = json.loads(request.data.decode())
    text = data['text']
    print(text)
    res_array, emotions = call_model(text)
    data_set = {'User': f'{user_query}', 'result':int(res_array[0]), 'emotion':emotions[0], 'Timestamp':time.time()}
    json_dump = json.dumps(data_set)

    return json_dump

if __name__ == '__main__':
    app.run(port=7778)