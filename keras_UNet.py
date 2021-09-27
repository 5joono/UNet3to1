import numpy as np
from scipy import io
import h5py
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, load_model, model_from_json
from keras.layers import Input, Lambda, BatchNormalization, Activation, Conv2D, add, MaxPooling2D, UpSampling2D, concatenate
from keras.applications.vgg16 import VGG16

from keras import backend as K
from keras.engine.topology import Layer
from keras import losses

def ResNet24(k):
    x_in = Input(shape=(None,None,1), name='input')

    x_0 = Conv2D(k, (3,3), padding='same', name = 'conv0')(x_in)

    # residual block 1
    x = BatchNormalization()(x_0)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_2')(x)
    x_1 = add([x,x_0]) # Skip connection


    # residual block 2
    x = BatchNormalization()(x_1)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv2_2')(x)
    x_2 = add([x,x_1]) # Skip connection


    # residual block 3
    x = BatchNormalization()(x_2)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv3_2')(x)
    x_3 = add([x,x_2]) # Skip connection

    # residual block 4
    x = BatchNormalization()(x_3)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv4_2')(x)
    x_4 = add([x,x_3]) # Skip connection

    # residual block 5
    x = BatchNormalization()(x_4)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv5_2')(x)
    x_5 = add([x,x_4]) # Skip connection

    # residual block 6
    x = BatchNormalization()(x_5)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv6_2')(x)
    x_6 = add([x,x_5]) # Skip connection

    # residual block 7
    x = BatchNormalization()(x_6)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv7_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv7_2')(x)
    x_7 = add([x,x_6]) # Skip connection

    # residual block 8
    x = BatchNormalization()(x_7)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv8_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv8_2')(x)
    x_8 = add([x,x_7]) # Skip connection

    # residual block 9
    x = BatchNormalization()(x_8)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_2')(x)
    x_9 = add([x,x_8]) # Skip connection

    # residual block 10
    x = BatchNormalization()(x_9)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv10_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv10_2')(x)
    x_10 = add([x,x_9]) # Skip connection

    # residual block 11
    x = BatchNormalization()(x_10)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv11_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv11_2')(x)
    x_11 = add([x,x_10]) # Skip connection

    # Aggregation
    x = BatchNormalization()(x_11)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv12')(x)

    # Residual(i.e. Noise) Learning
    x = add([x,x_in], name = 'output')

    model = Model(inputs=x_in, outputs=x)

    return model

def UNet24_ver2(k):

    x_in = Input(shape=(None,None,1), name='input')
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_1')(x_in)

    # Module 1
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv1_2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x1 = Conv2D(k, (3,3), padding='same', name = 'conv1_3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x1)

    # Module 2
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv2_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x2 = Conv2D(2*k, (3,3), padding='same', name = 'conv2_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x2)

    # Module 3
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv3_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x3 = Conv2D(4*k, (3,3), padding='same', name = 'conv3_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x3)

    # Module 4
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv4_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    skip_x4 = Conv2D(8*k, (3,3), padding='same', name = 'conv4_2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(skip_x4)

    # Module 5
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16*k, (3,3), padding='same', name = 'conv5_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16*k, (3,3), padding='same', name = 'conv5_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv6_0')(x)
    # x = Conv2DTranspose(8*k, (3,3), strides=(2, 2), padding='same', name = 'upconv6_1')(x)

    x = concatenate([x,skip_x4], axis = -1)

    # Module 6
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv6_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(8*k, (3,3), padding='same', name = 'conv6_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv7_0')(x)
    # x = Conv2DTranspose(4*k, (3,3), strides=(2, 2), padding='same', name = 'upconv7_1')(x)


    x = concatenate([x,skip_x3], axis = -1)

    # Module 7
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv7_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4*k, (3,3), padding='same', name = 'conv7_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv8_0')(x)
    # x = Conv2DTranspose(2*k, (3,3), strides=(2, 2), padding='same', name = 'upconv8_1')(x)


    x = concatenate([x,skip_x2], axis = -1)

    # Module 8
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv8_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(2*k, (3,3), padding='same', name = 'conv8_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_0')(x)
    # x = Conv2DTranspose(k, (3,3), strides=(2, 2), padding='same', name = 'upconv9_1')(x)


    x = concatenate([x,skip_x1], axis = -1)

    # Module 9
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(k, (3,3), padding='same', name = 'conv9_2')(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(1, (1,1), padding='same', name = 'conv9_3')(x)

    x = add([x,x_in], name = 'output') # Residual Learning
    model = Model(inputs=x_in, outputs=x)

    return model

batch_size = 4
epoch_to_start = 0
epochs = 30
data_augmentation = False
fine_tuning = False

learning_rate = 0.0001

model_name = 'UNet_CBCT_ver1'
save_model_address = 'D:/MODELS/CTmeeting2018_talk/model_'+model_name+'.h5'
save_history_address = 'D:/MODELS/CTmeeting2018_talk/model_'+model_name

load_model_name = 'UNet_ver10_2'
load_model_address = 'D:/MODELS/CTmeeting2018_talk/model_'+load_model_name+'.h5'




opt = keras.optimizers.Adam(lr=learning_rate)

if fine_tuning:
    model = load_model(load_model_address)

    opt = keras.optimizers.Adam(lr=learning_rate)

    model.compile(optimizer=opt, loss=['mean_squared_error'], metrics = ['mean_absolute_error'])


    model.summary()

else:
    model = UNet24_ver2(64)

    model.compile(optimizer=opt, loss=['mean_squared_error'], metrics = ['mean_absolute_error'])

    model.summary()



matf = io.loadmat('D:/DATASET/result.mat')
X_temp = matf['Origin']
X = X_temp[:,:,0:2400].T
X_val = X_temp[:,:,2400:3000].T

Y_temp = matf['Cone']
Y = Y_temp[:,:,0:2400].T
Y_val = Y_temp[:,:,2400:3000].T

num_img = Y.shape[0]
num_val_img = Y_val.shape[0]

X = X.reshape(num_img,256,256,-1)
Y = Y.reshape(num_img,256,256,-1)

X_val = X_val.reshape(num_val_img,256,256,-1)
Y_val = Y_val.reshape(num_val_img,256,256,-1)


import math
from keras.callbacks import LearningRateScheduler, Callback, TensorBoard

class BatchLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses_mse = []
        self.losses_mae = []
        self.losses_output_mse = []
        self.losses_output_mae = []
        self.losses_reg = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.losses_mse.append(logs.get('mean_squared_error'))
        self.losses_mae.append(logs.get('mean_absolute_error'))
        self.losses_output_mse.append(logs.get('output_mean_squared_error'))
        self.losses_output_mae.append(logs.get('output_mean_absolute_error'))
        self.losses_reg.append(logs.get('custom_regularizer_2_mean_absolute_error'))


batch_history = BatchLossHistory()

tb_hist = TensorBoard(log_dir='D:/log', histogram_freq=0, write_graph=True, write_images=True)

check_point = keras.callbacks.ModelCheckpoint(save_model_address, monitor='loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

callback_list = [check_point, batch_history, tb_hist]

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(Y[:,:,:,:], X[:,:,:,:],
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True,
              initial_epoch=epoch_to_start,
              callbacks = callback_list,
              verbose = 1,
              validation_data = (Y_val[:,:,:,:], X_val[:,:,:,:])
                        )
else:
    print('Using real-time data augmentation.')

    data_gen_args = dict(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=60,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range=0,
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 10
    image_datagen.fit(Y, augment=True, seed=seed)
    mask_datagen.fit(X, augment=True, seed=seed)

    image_generator = image_datagen.flow(Y, batch_size=batch_size, seed=seed)
    mask_generator = mask_datagen.flow(X, batch_size=batch_size, seed=seed)

    train_generator = zip(image_generator, mask_generator)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=Y.shape[0] // batch_size,
        epochs=epochs, initial_epoch=epoch_to_start,
        callbacks = callback_list,
        validation_data = (Y_val, X_val))

np.save(save_history_address, history.history)

io.savemat(save_history_address, mdict={
                                        'loss': batch_history.losses,
#                                         'mean_squared_error': batch_history.losses_mse,
                                        'mean_absolute_error': batch_history.losses_mae,
#                                         'output_mean_squared_error': batch_history.losses_output_mse,
#                                         'output_mean_absolute_error': batch_history.losses_output_mae,
#                                         'reg_loss':  batch_history.losses_reg
                                       }
          )



# model_name = 'UNet_CBCT_ver1'
# model_address = 'D:/MODELS/CTmeeting2018_talk/model_'+model_name+'.h5'
# model = load_model(model_address)

# matf = io.loadmat('D:/DATASET/result.mat')
# X_temp = matf['Origin']
# X_val = X_temp[:,:,2400:3000].T

# Y_temp = matf['Cone']
# Y_val = Y_temp[:,:,2400:3000].T

# num_val_img = Y_val.shape[0]

# X_val = X_val.reshape(num_val_img,256,256,-1)
# Y_val = Y_val.reshape(num_val_img,256,256,-1)

# Check model:
model.summary()

output_4d = model.predict(Y_val, batch_size=4)
# model.evaluate(Y_val, batch_size=1)

### File address of which the results will be saved, e.g.,
results_address = 'D:/MODELS/CBCT/'+model_name+'_output.mat'

### Save the results
io.savemat(results_address, mdict={'output': output_4d[:,:,:,0].T,'input': Y_val[:,:,:,0].T,'ref': X_val[:,:,:,0].T})
