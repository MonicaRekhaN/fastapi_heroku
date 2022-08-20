from flask import Flask, render_template, request

import cv2
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
from http.client import HTTPResponse
#from urllib import request
import uvicorn
#from fastapi import FastAPI, File, UploadFile,Request
import cv2
from keras.models import Model
from starlette.responses import RedirectResponse
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.optimizers import Adam
from fastapi.templating import Jinja2Templates
from keras.models import Sequential, Model
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
#from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from app import *
from functions import Transformer, create_masks_decoder, evaluate, load_image
from pickle import load,dump
from starlette.responses import Response
# from starlette.templating import Jinja2Templates as _Jinja2Templates, _TemplateResponse


   
num_layer = 4
d_model = 512
dff = 2048
num_heads = 8
row_size = 7
col_size = 7
top_k=5000
target_vocab_size = top_k + 1 # top_k = 5000
dropout_rate = 0.1



#Building a Word embedding for top 5000 words in the captions
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
pkl_tokenizer_file=r"C:\Model Deployment\New folder\ImageCaptioning-FastAPI-WIP\encoded_tokenizer.pkl"
# Load the tokenizer train features to disk
with open(pkl_tokenizer_file, "rb") as encoded_pickle:
    tokenizer = load(encoded_pickle)
#Image Model
image_model = ResNet50(include_top=False,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
new_input = image_model.input
hidden_layer = image_model.layers[-2].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
transformer = Transformer(num_layer,d_model,num_heads,dff,row_size,
                col_size,target_vocab_size,max_pos_encoding=target_vocab_size,rate=dropout_rate)
# transformer()
start_token = tokenizer.word_index['<start>']
end_token = tokenizer.word_index['<end>']
decoder_input = [start_token]
output = tf.expand_dims(decoder_input, 0) #token
dec_mask = create_masks_decoder(output)
test = tf.random.Generator.from_seed(123)
test = test.normal(shape=(16,49,2048))
transformer(test,output,False,dec_mask)
# transformer.load_weights('model.h5')
# transformer.built=True
# transformer.built=True
transformer.load_weights('model.h5')

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after', methods=['GET', 'POST'])
def after():
    global model, resnet, vocab, inv_vocab

    img = request.files['file1']

    img.save(r'static\file.jpg')

    
    # image = cv2.imread(r'C:\Model Deployment\New folder\image-captioning-keras-resnet-main\static\file.jpg')
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, (224,224))
    # image = np.reshape(image, (1,224,224,3))
    image_path = r'static\file.jpg'
    
    #image = tf.io.read_file(r'C:\Model Deployment\New folder\image-captioning-keras-resnet-main\static\file.jpg')
    # image = tf.image.decode_jpeg(image, channels=3)
    # image = tf.image.resize(image, (224, 224))
    
    caption,result,attention_weights = evaluate(image_path,transformer)

    #remove "<unk>" in result
    for i in caption:
        if i=="<unk>":
            caption.remove(i)

    #remove <end> from result         
    result_join = ' '.join(caption)
    result_final = result_join.rsplit(' ', 1)[0]
    return render_template('after.html', data=' '.join(caption))
    #return templates.TemplateResponse("after.html",{"data":result})


# if __name__ == "__main__":
#     uvicorn.run(app, debug=True)

if __name__ == "__main__":
    app.run(debug=True,port=80)


