# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 17:44:29 2022

@author: Mahan HP
Source: https://github.com/keras-team/keras-io/blob/e7bd2163ace8ea44d0487dad7f7416ff933adb79/examples/vision/ipynb/pointnet.ipynb
"""

import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split

current_folder = Path.cwd()

DATA_DIR = os.path.join((current_folder), "ModelNet40")

print (DATA_DIR)

def parse_dataset(num_points,DATA_DIR):

    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "*"))
    print(folders)
    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("\\")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

NUM_POINTS = 2048
NUM_CLASSES = 10
BATCH_SIZE = 32

train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS,DATA_DIR)
print (test_points)





# =============================================================================
# def build_model_01():
#     model = keras.Sequential([
#         #Input layer
#         keras.layers.Flatten(input_shape=( 2048, 3)),
#         
#         #Hidden layers
#         keras.layers.Dense(units=1000, activation='relu'),
#         keras.layers.Dense(units=500, activation='relu'),
#         keras.layers.Dense(units=100, activation='relu'),
#         
#         #Output layer
#         keras.layers.Dense(units=5, activation='softmax')
#         ])
# =============================================================================
    
# =============================================================================
#     
# def build_model_cnn():
#     model = keras.Sequential([
#                 
#         #CONVOLUTION LAYERS
#         #Convolution 01
#         keras.layers.Conv1D(filters=32, kernel_size=(1), padding="valid", activation='relu', input_shape = (2048, 3))
#         #keras.layers.MaxPool1D(pool_size=(2)),
# =============================================================================
        
# =============================================================================
#         #Convolution 02
#         keras.layers.Conv1D(filters=128, kernel_size=(3), activation='relu'),
#         keras.layers.MaxPool1D(pool_size=(2)),
#         
#         #Convolution 02
#         keras.layers.Conv1D(filters=256, kernel_size=(3), activation='relu'),
#         keras.layers.MaxPool1D(pool_size=(2)),
# =============================================================================
        
        
        #FULLY CONNECTED NETWORK
# =============================================================================
#         #Input layer
#         keras.layers.Flatten(),
#         
#         #Hidden layers
#         keras.layers.Dense(units=100, activation='relu'),
#         
#         #Output layer
#         keras.layers.Dense(units=2, activation='softmax')
# =============================================================================
#        ])    
    




def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))


train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)


def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.2):
    assert (train_split + val_split) == 1
    #assert statement is used to continue the execute if the given condition evaluates to True.
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_dataset = ds.take(ds_size)    
    val_dataset = ds.skip(train_size).take(val_size)
    
    return train_dataset, val_dataset

train_dataset, val_dataset = get_dataset_partitions_tf(train_dataset, len(train_dataset))


# =============================================================================
# train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
# val_dataset = val_dataset.shuffle(len(train_points)*0.2).map(augment).batch(BATCH_SIZE)
# =============================================================================

# =============================================================================
# 
# #Architecture 
# 
# =============================================================================


# =============================================================================
# class OrthogonalRegularizer(keras.regularizers.Regularizer):
#     def __init__(self, num_features, l2reg=0.001):
#         self.num_features = num_features
#         self.l2reg = l2reg
#         self.eye = tf.eye(num_features)
# 
#     def __call__(self, x):
#         x = tf.reshape(x, (-1, self.num_features, self.num_features))
#         xxt = tf.tensordot(x, x, axes=(2, 2))
#         xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
#         return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
#     
#     
# =============================================================================
    
    
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)


def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)



def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    # reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        # activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])







inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)

# x = conv_bn(inputs, 32)

x = conv_bn(x, 32)
# x = tnet(x, 64)
#x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 128)
x = conv_bn(x, 1024)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 512)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)


outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
current_model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
#summary
current_model.summary()



#return model


#inputs = keras.Input(shape=(NUM_POINTS, 3))
#model
#current_model = build_model_cnn()




#train_points, test_points , train_labels, test_labels = train_test_split(train_points, train_labels, test_size=0.2, random_state=40)
current_model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["sparse_categorical_accuracy"]
)

model_history = current_model.fit(train_dataset, batch_size=BATCH_SIZE, epochs=100, validation_data=val_dataset)

plt.figure()
plt.plot(model_history.history['sparse_categorical_accuracy'], label='sparse_categorical_accuracy')
plt.plot(model_history.history['val_sparse_categorical_accuracy'], label='val_sparse_categorical_accuracy')
# =============================================================================
# plt.plot(model_history.history['loss'], label='loss')
# plt.plot(model_history.history['val_loss'], label='val_loss')
# =============================================================================
plt.xlabel("epoch")
plt.ylabel('accuracy')
plt.ylim([0, 1])


current_model.save(r'.\saved_model\current_model_modelnet40-tnet64-10class-100epochs')

#load model
model_saved = keras.models.load_model(r'.\saved_model\current_model_modelnet40-tnet64-10class-100epochs')
model_saved.summary()

current_model = model_saved


# =============================================================================
# mesh = trimesh.load("C:/Users/Mahan HP/Desktop/S9_pointcloud_classification/test-models-photogrammetry/monitor_0467.off")
# mesh.show()
# points = mesh.sample(2048)
# print(points)
# fig = plt.figure(figsize=(5, 5))
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(points[:, 0], points[:, 1], points[:, 2],s=1, c='black')
# ax.set_axis_off()
# plt.show()
# 
# sample = np.reshape(points, (1,2048,3))
# preds = current_model.predict(sample)
# print(preds)
# preds = tf.math.argmax(preds, -1)
# print(preds)
# #preds = np.argmax(preds)
# print (preds.numpy()[0])
# print("pred: {:}".format(CLASS_MAP[preds.numpy()[0]]))
# 
# fig = plt.figure(figsize=(15, 10))
# for i in range(1):
#     ax = fig.add_subplot(1, 1, i + 1, projection="3d")
#     ax.scatter((np.asarray(points))[:,0], (np.asarray(points))[:,1], (np.asarray(points))[:,2])
#     ax.set_title(
#         "pred: {:}, label: {:}".format(CLASS_MAP[preds.numpy()[0]], CLASS_MAP[3])
#     )
#     ax.set_axis_off()
# plt.show()
# =============================================================================


#Load Photogrammetry or other models for prediction
#PREDS_DIR = os.path.join((current_folder), "mixed test data")
PREDS_DIR = os.path.join((current_folder), "Test data modelnet 40 10 classes")
#PREDS_DIR = os.path.join((current_folder), "Photogrammetry download")
PREDS_DIR = glob.glob(os.path.join(PREDS_DIR, "*"))
mesh = []
for path in PREDS_DIR:
    mesh.append(trimesh.load(path))
#mesh = trimesh.load("C:/Users/Mahan HP/Desktop/S9_pointcloud_classification/test-models-photogrammetry/monitor_0467.off")
# mesh.show()

pred_files = []
for m in mesh:    
    #pred_files.append(np.reshape(m.sample(2048), (1,2048,3)))
    pred_files.append(m.sample(2048))

pred_files = np.array(pred_files)

#pred_files = list(pred_files)[0]


# =============================================================================
# pred_files = (np.reshape(mesh.sample(2048), (1,2048,3)))
# =============================================================================
#
# =============================================================================
# preds = []
# for p in pred_files:
#     
#     pred = current_model.predict(p)
#     pred = tf.math.argmax(pred, -1)
#     preds.append(pred)
# 
# =============================================================================
preds = current_model.predict(pred_files)
print(preds)
preds = tf.math.argmax(preds, -1)

print(preds)

# plot points with predicted class and label
fig = plt.figure(figsize=(15, 10))
for i in range(20):
    ax = fig.add_subplot(5, 4, i + 1, projection="3d")
    ax.scatter(pred_files[i, :, 0], pred_files[i, :, 1], pred_files[i, :, 2],s=1, c='black')
    ax.set_title(
        "pred: {:}".format(
            CLASS_MAP[preds[i].numpy()]
        )
    )
    ax.set_axis_off()
plt.show()



