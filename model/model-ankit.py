import data_playground
import tensorflow as tf
import numpy as np
from tensorflow.python import debug as tf_debug
from sklearn.preprocessing import StandardScaler


(data_pre, labels_pre) = data_playground.create_data_set()


assert len(data_pre) == len(labels_pre)

#shuffles the data and labels
data_and_labels = np.hstack([data_pre, labels_pre])
np.random.shuffle(data_and_labels)

data = data_and_labels.T[:10].T
labels_dx_dy = data_and_labels.T[11:].T
labels_category = data_and_labels.T[10].T


random_noise = np.random.randn(data.shape[0], data.shape[1]) / 20
#data = data + random_noise

sc = StandardScaler()
data = sc.fit_transform(data)
labels_dx_dy = sc.fit_transform(labels_dx_dy)

random_noise = np.random.randn(data.shape[0], data.shape[1]) / 2

#adds random noise to the data to ensure that the model is robust
data = data + random_noise

data = sc.fit_transform(data)

x_train  = data[:-100]
x_test = data[-100:]

y_train = labels_category[:-100].reshape((len(labels_category[:-100]), 1))
y_test = labels_category[-100:].reshape((100, 1))




print(x_train[:5])
print(y_train[:5])

print(y_train.shape)
print(x_train.shape)



inputs= tf.keras.Input(shape = (10,))
x = tf.keras.layers.Dense(10, activation = 'relu')(inputs)
x = tf.keras.layers.Dropout(0.1)(x)
x = tf.keras.layers.Dense(8, activation = 'relu')(x)
x=tf.keras.layers.Dropout(0.1)(x)
outputs = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
model_category = tf.keras.Model(inputs = inputs, outputs = outputs)
model_category.summary()
model_category.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
#model_category.fit(x_train, y_train, epochs=70, batch_size=20, validation_split=0.01)
#print(model_category.evaluate(x_test, y_test))



data_positive = np.array([data[i] for i in range(len(data)) if labels_category[i] == 1])
labels_dx_dy_positive = np.array([labels_dx_dy[i] for i in range(len(data)) if labels_category[i] == 1])

x_dx_dy_train = data_positive[:-100]
y_dx_dy_train = labels_dx_dy_positive[:-100]

x_dx_dy_test = data_positive[-100:]
y_dx_dy_test = labels_dx_dy_positive[-100:]




inputs = tf.keras.Input(shape = (10,))
outputs = tf.keras.layers.Dense(2, activation= 'linear', kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
model= tf.keras.Model(inputs = inputs, outputs= outputs)
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error', metrics=['mean_squared_error'])

model.fit(x_dx_dy_train, y_dx_dy_train, epochs=50, batch_size = 20, validation_split = 0.01)
print(model.evaluate(x_dx_dy_test, y_dx_dy_test))



