from pyexpat import model
import uvicorn
from fastapi import FastAPI,Request,Form,UploadFile,File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.datastructures import URL
import base64
import pickle
from PIL import Image
from io import BytesIO
import glob
from pickle import load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras import optimizers
from keras import Input, layers
from keras.preprocessing import image
from keras.models import Model
import cv2
from base64 import b64encode
import io
from keras.models import load_model
import numpy as np
from keras.applications import ResNet50
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.layers import add
from keras.utils import np_utils
from keras.preprocessing import image, sequence
import cv2
from keras_preprocessing.sequence import pad_sequences
#from keras.preprocessing.image import load_img, img_to_array
from keras.utils.image_utils import img_to_array,load_img

app = FastAPI()
global resnet, vocab, inv_vocab

# app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")
templates.env.globals['URL'] = URL

def preprocess(image_path):
    # Convert all the images to size 224X224 as expected by the inception v3 model
    print(image_path)
    img = load_img(image_path, target_size=(224, 224,3))
    
    # Convert PIL image to numpy array of 3-dimensions
    x = img_to_array(img)
    #print(x.shape())
    # Add one more dimension
    x = np.expand_dims(x, axis=0)
    # preprocess the images using preprocess_input() from inception module
    #x = preprocess_input(x)
    return x
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = resnet1.predict(image) # Get the encoding vector for the image
    print("testing")
    print(fea_vec.shape)
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec


with open('wordtoix.pkl', "rb") as encoded_pickle:
    wordtoix = load(encoded_pickle)
with open('ixtoword.pkl', "rb") as encoded_pickle:
    ixtoword = load(encoded_pickle)



max_length = 34
vocab_size = 1652
embedding_dim = 200

async def loadModel():
    #Loading LSTM mddel
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256,return_sequences =True)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = LSTM(256)(decoder1)
    decoder3 = Dense(256, activation='relu')(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)
    model1 = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model1.load_weights('model.h5')
    return model1
# with open('model.pkl', "rb") as encoded_pickle:
#     model = load(encoded_pickle)

# resnet = pickle.load(open('resnet_model.pkl', 'rb'))
resnet = ResNet50(include_top=True,weights='imagenet',input_shape=(224, 224,3),pooling="avg")
resnet1 = Model(resnet.input, resnet.layers[-2].output) 

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/form",response_class=HTMLResponse)
def form_get(request: Request):
    print("testing")
    global model
    model=loadModel()
    return templates.TemplateResponse('form.html', context={'request': request})

@app.post("/after",response_class=HTMLResponse)
async def form_post(request: Request,file: UploadFile = File(...)):
    global model
    print(file.filename)
    
    print("in predict")
    f = request.files['file1'].read()
    buf = BytesIO(f)
    npimg = np.fromstring(f,np.uint8)
    img = cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.getvalue()).decode('ascii')
    mime = "image/jpeg"
    uri = "data:%s;base64,%s"%(mime, img_base64)
#contents = file.file.read()
#base64_encoded_image = base64.b64encode(contents).decode("utf-8")
    model= await model
    image = encode(buf).reshape((1,2048))
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        #print(model)
        yhat = model.predict([image,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return templates.TemplateResponse('after.html', context={'request': request,'data': final,'img_data' : uri})


 # at last, the bottom of the file/module
if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=5000)
    uvicorn.run(app, host="0.0.0.0", port=8080)
