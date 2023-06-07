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

PREPROCESS_MODEL = "bert_en_uncased_preprocess_3"
BERT_MODEL = 'TF2'

transformer_encoder = hub.KerasLayer(BERT_MODEL, trainable=False)
preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
# inp = preprocessor(["hello"])
# print(inp)

class TextClassificationModel(tf.keras.Model):
    def __init__(self, PREPROCESS_MODEL, BERT_MODEL):
        super().__init__()
        self._preprocessor = hub.KerasLayer(PREPROCESS_MODEL)
        self._transformer_encoder = hub.KerasLayer(BERT_MODEL, trainable=True)
        self._hidden_layer1 = tf.keras.layers.Dense(units=512, activation='relu')
        self._hidden_layer2 = tf.keras.layers.Dense(units=256, activation='relu')
        self._hidden_layer3 = tf.keras.layers.Dense(units=128, activation='relu')
        self._classifier = tf.keras.layers.Dense(units=8, activation='softmax')

    def call(self, inputs):
        preprocessed_inputs = self._preprocessor(inputs)
        embedding = self._transformer_encoder(preprocessed_inputs)['pooled_output']
        hidden1_output = self._hidden_layer1(embedding)
        hidden2_output = self._hidden_layer2(hidden1_output)
        hidden3_output = self._hidden_layer3(hidden2_output)
        logits = self._classifier(hidden3_output)
        return logits



model = TextClassificationModel(PREPROCESS_MODEL, BERT_MODEL)

init_lr = 3e-5
optimizer = tf.keras.optimizers.Adam(init_lr)

model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=optimizer,
    metrics=[tf.metrics.CategoricalAccuracy()]
)


# Restore the weights
model.load_weights('Weights')

def call_model(text):
  # inputs = json.loads(text)
  # print(inputs)
  results = model.call(text)
  # print(results)
  res_array = np.argmax(results, axis=1)
  res_array
  res = []
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
      res.append('surprise')
    elif (test_value == 5):
      res.append('love')
    elif (test_value == 6):
        res.append('fear')
    else:
      res.append('disgust')
  # print(res)
  return res_array, res, results

def sort_ids(descriptions, post_ids, chat):
  res_array, emotions, description_metrices = call_model(descriptions)
  chat_res_array, chat_emotions, res_metrix_chat = call_model(chat)
  corr_coefficient_list = []
  for i in description_metrices:
      correlation_matrix = np.corrcoef(res_metrix_chat, i)

      # Access the correlation coefficient value
      correlation_coefficient = correlation_matrix[0, 1]

      corr_coefficient_list.append(correlation_coefficient)

  # Create a dictionary with post_ids as keys and corr_coefficient_list as values
  dictionary = {post_ids[i]: corr_coefficient_list[i] for i in range(len(post_ids))}

  # Sort the lists based on the coefficients
  sorted_data = sorted(zip(post_ids, corr_coefficient_list), key=lambda x: x[1], reverse=True)

  # Retrieve the sorted post_ids and coefficients as separate lists
  sorted_ids, sorted_coefficients = zip(*sorted_data)


  return sorted_ids, chat_emotions, chat_res_array

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
    descriptions = data['description_list']
    post_ids = data["id_list"]
    chat = data["chat"]
    sorted_ids, chat_res, chat_emotion = sort_ids(descriptions, post_ids, chat)

    chat_emotion = json.dumps(np.array(chat_emotion).tolist())
    # data_set = {'User': f'{user_query}', 'result':res_array,"res_metrix":json_res_metrix, 'emotion':emotions, 'Timestamp':time.time()}

    data_set = {'User': f'{user_query}', 'sorted_ids': sorted_ids, "chat_result": chat_res, 'chat_emotion': chat_emotion,
                'Timestamp': time.time()}

    print(data_set)

    json_dump = json.dumps(data_set)

    return json_dump

if __name__ == '__main__':
    app.run(port=7778)