import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras

# It can be used to reconstruct the model identically.
reconstructed_model = keras.models.load_model("my_model")

test_input = np.random.random((128, 32))
test_target = np.random.random((128, 1))

# The reconstructed model is already compiled and has retained the optimizer
# state, so training can resume:
reconstructed_model.fit(test_input, test_target)

testx = np.random.random((10, 32))
testy = np.random.random((10, 1))

#  print('Accuracy = ', reconstructed_model.score(testx, testy))

predictions = reconstructed_model.predict(testx)
for x in range(10):
    print('\nPredicted Final Note:', predictions[x], '\nFactors:', testx[x], '\nActual Final Note:', testy[x])

