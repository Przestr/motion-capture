import tensorflow as tf 
from tensorflow.keras.models import load_model
import pickle
import numpy as np 

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

x  = []

model = load_model("weights-improvement-60-0.0663-biggeer.hdf5", compile=False)

pickle_in = open('C:/Users/Strum/Desktop/cvapr/Movement-classification/pickle_data/09_01_worldpos.pickle',"rb")
data = pickle.load(pickle_in)

data = data[:100]
data = np.array(data)
data = np.reshape(data, (1, data.shape[0],data.shape[1]*data.shape[2]))

result = []
result = model.predict(data)


if(result[0][0] == 1.0):
    print("Result of the prediction: walk")
elif(result[0][1] == 1.0):
    print("Result of the prediction: run")
else:
    print("Result of the prediction: jump")


#print(model.predict(data))

