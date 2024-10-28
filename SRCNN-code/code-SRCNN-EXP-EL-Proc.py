import numpy as np
import tensorflow as tf
import os
from netCDF4 import Dataset
import cv2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.losses import mean_squared_error, KLDivergence
import keras.backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation,Conv2D, MaxPooling2D, Input, ZeroPadding2D, add, Conv2DTranspose, PReLU, Dropout
from timeit import default_timer as timer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision
exp = "SRCNN-EXP-EL-Proc"
class TimingCallback(keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs=[]
    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = timer()
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(timer()-self.starttime)


def read_WRF(tt,var):
    fil        = Dataset(f'WRF_CONUS_1980-2019_daily.nc')
    hr_var     = fil.variables[f'{var}'][0:tt,:,:]
    return hr_var

def read_ERA5(tt1,tt2,var):
    fil        = Dataset(f'ERA5_CONUS_1980-2019_daily_processed.nc')
    temp       = fil.variables[f'{var}'][tt1:tt2,:,:]
    temp1      = temp[:,:,:]
    temp1      = np.where(temp1 < 0,0,temp1)
    lr_var     = temp1*1000
    return lr_var


def read_elev(tt):
    felev      = Dataset(f'HGT_WRF_CONUS.nc')
    elev1      = felev.variables["HGT"]
    elev       = np.tile(elev1,(tt,1,1))
    return elev

def minmaxscaler(lr):
    tt       = np.shape(lr)[0]
    nx       = np.shape(lr)[1]
    ny       = np.shape(lr)[2]
    scaler   = MinMaxScaler()
    lr       = lr.flatten()
    lr       = scaler.fit_transform(lr.reshape(-1,1))
    lr       = np.reshape(lr,(tt,nx,ny,1))
    return lr


def split(lr,hr):
    X_train, X_test, y_train, y_test = train_test_split(lr, hr, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

def exp_loss(y_true, y_pred):
    epsilon = 1e-15  # Small constant to avoid log(0)
    y_pred = K.clip(y_pred, epsilon, 1 - epsilon)
    w=3
    exp_true = K.exp(w*y_true)
    exp_pred = K.exp(w*y_pred)
    # Calculate` the mean squared error of the exponentials
    loss = K.mean(K.square(exp_true - exp_pred), axis=-1)
    return loss

def quantile_loss(q, y, y_p):
        e = y-y_p
        return tf.keras.backend.mean(tf.keras.backend.maximum(q*e, (q-1)*e))

def model_fit(X_train ,y_train,X_test,y_test):
    inshp   = np.shape(X_train)
    model   = Sequential()
    # Layer 1
    model.add(Conv2D(64, kernel_size=(9, 9), activation='relu', padding='same', input_shape=(inshp[1],inshp[2],inshp[3])))

    # Layer 2
    model.add(Conv2D(32, kernel_size=(1, 1), activation='relu', padding='same'))

    # Layer 3
    model.add(Conv2D(1, kernel_size=(5, 5), activation='relu', padding='same'))
    model.summary()
    cb             = TimingCallback()
    optimizer      = Adam(lr=0.0001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    model.compile(optimizer=optimizer, loss=exp_loss, metrics=['mean_squared_error'],run_eagerly=True)
    history        = model.fit(X_train ,y_train, validation_split=0.1,batch_size=10,epochs=100,shuffle=True,callbacks=[cb,early_stopping])
    print(history.history.keys())
    train_loss     = history.history['loss']
    val_loss       = history.history['val_loss']
    time           = cb.logs
    np.save(f'./train_loss_daily_{exp}.npy',train_loss)
    np.save(f'./val_loss_daily_{exp}.npy',val_loss)
    np.save(f'./time_daily_{exp}.npy',time)
    model.save(f'./my_model_daily_{exp}.h5')
    return model 

def predict(X,model):
    ypred   = model.predict(X)
    return ypred

def invtrans_write(y,scalar,name,exp):
    shp    = np.shape(y)
    tt     = shp[0]
    nhr1   = shp[1]
    nhr2   = shp[2]
    y      = y.flatten()
    yinv   = scalar.inverse_transform(y.reshape(-1, 1))
    yinv   = np.reshape(yinv,(tt,nhr1,nhr2,1))
    np.save(f'./{name}_daily_{exp}.npy',yinv)

def main():
    nyear = 30
    tt30  = 365*30 + 8
    tt40  = 365*40 + 10
    tt    = tt30
    nhr1  = 216
    nhr2  = 488
    nlr1  = 216
    nlr2  = 488
# Read data 
    hr_prect        = read_WRF(tt,"RAIN") 
    lr_prect        = read_ERA5(0,tt40,"tp")
# Scale high-resolution precipitation ("y")
    scaler_hrprect  = MinMaxScaler()
    hr_prect        = hr_prect.flatten()
    hr_prect_scaled = scaler_hrprect.fit_transform(hr_prect.reshape(-1,1))
    hr_prect_scaled = np.reshape(hr_prect_scaled,(tt,nhr1,nhr2,1))
    
# Scale low-resolution precipitation ("x")
    lr_prect        = lr_prect.flatten()
    lr_prect_scaled = scaler_hrprect.transform(lr_prect.reshape(-1, 1))
    lr_prect_scaled = np.reshape(lr_prect_scaled,(tt40,nlr1,nlr2,1))
    elev            = read_elev(tt40)
    elev_scaled     = 0.05*minmaxscaler(elev)
    lr              = np.concatenate((lr,elev_scaled[0:tt,:,:,:]),axis=3)
#Final high-res (y) and low-res data (x)
    hr              = hr_prect_scaled
    X_train, X_test, y_train, y_test = split(lr,hr)

    os.environ["CUDA_VISIBLE_DEVICES"] = '1' #use GPU with ID=1
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
    config.gpu_options.allow_growth = True #allocate dynamically
    sess   = tf.compat.v1.Session(config=config) 
    tf.compat.v1.keras.backend.set_session(sess)
    print(tf.config.experimental.list_physical_devices('GPU'))
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    model          = model_fit(X_train[:,:,:,:] ,y_train[:,:,:,0:1],X_test[:,:,:,:],y_test[:,:,:,0:1])
    X_val4         = lr_prect_scaled[tt30:tt40,:,:,:]
    y_val4_predict = predict(X_val4,model)
    y_test_predict = predict(X_test,model)
    invtrans_write(y_val4_predict,scaler_hrprect,"y_val4_predict",exp)
    invtrans_write(y_test_predict,scaler_hrprect,"y_test_predict",exp)
    invtrans_write(y_test[:,:,:,0],scaler_hrprect,"y_test",exp)
    invtrans_write(X_test[:,:,:,0],scaler_hrprect,"X_test",exp)

if __name__ == "__main__":
    main()
