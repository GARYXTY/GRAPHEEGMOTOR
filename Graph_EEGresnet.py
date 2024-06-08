from tensorflow.keras.layers import Conv2D, Layer, InputSpec, Input, Dense, Conv1D, MaxPooling1D, Flatten, Activation, add, Dropout, concatenate, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import regularizers
import os
from keras_dgl.layers import MultiGraphCNN
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, multiply, Permute, RepeatVector
from tensorflow.keras import backend as K
import tensorflow as tf

def mycrossentropy(y_true,y_pred,e=0):
	loss1 = tf.keras.losses.CategoricalCrossentropy()(y_true,y_pred)
	loss2 = tf.keras.losses.CategoricalCrossentropy()(K.ones_like(y_pred)/4,y_pred)
	return (1-e)*loss1 + e*loss2

def mycrossentropy_wrapper(e):
    def loss(y_true, y_pred):
        return mycrossentropy(y_true, y_pred, e)
    return loss


if True:
    L = regularizers.l2(0.01)
else:
    L = None

def res_first(input_tensor,filters,kernel_size, fuse_type):
    eps=1.1e-5
    nb_filter1, nb_filter2 = filters
    x = Conv2D(filters=nb_filter1,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(input_tensor) ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(x) ##
    x, x_graph, power_feature = PSD_sum(activation = 'elu', data_format='channels_first')(x)
    x = add([x,input_tensor])
    x = Dropout(0.3)(x)
    return x, x_graph, power_feature


def res_subsam(input_tensor,filters,kernel_size,subsam, fuse_type):
    eps= 1.1e-5
    nb_filter1, nb_filter2 = filters
    x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=nb_filter1,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(x) ##
    x = MaxPooling2D((1, 2))(x)
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(x) ##	
    x = Dropout(0.3)(x)
    short = Conv2D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(input_tensor)
    short = MaxPooling2D((1, 2))(short)
    x, x_graph, power_feature = PSD_sum(activation = 'elu', data_format='channels_first')(x)
    x = add([x,short])
    return x, x_graph, power_feature


def res_nosub(input_tensor,filters,kernel_size, fuse_type):
    eps= 1.1e-5
    nb_filter1, nb_filter2 = filters
    x = BatchNormalization(epsilon=eps, axis=-1)(input_tensor)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=nb_filter1,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(x) ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)
    x = Conv2D(filters=nb_filter2,kernel_size=kernel_size,padding='same',use_bias=True, kernel_regularizer = L)(x) ##
    x, x_graph, power_feature = PSD_sum(activation = 'elu', data_format='channels_first')(x)	
    x = add([x,input_tensor])
    x = Dropout(0.3)(x)
    return x, x_graph, power_feature

def preprocess_adj(adj):
    (batch, Channel, _) = adj.shape
    batch_identity = tf.eye(Channel)
    adj = adj + batch_identity
    d = tf.math.reduce_sum(adj, axis=1)
    x = tf.constant([-0.5])
    d = tf.math.pow(d, x)
    d = tf.linalg.diag(d, k=0)
    adj_new = Permute(dims=(2,1))(adj*d) * d
    return adj_new


def PSD_sum(activation = "relu", data_format = 'channels_first', ki = "he_normal"):


    def f(input_x):

        channel_axis = -1 if data_format == 'channels_last' else 1
        
        (_, input_channels, length, dim) = input_x.shape


        input_x_2 = BatchNormalization()(input_x)
        input_x_2 = Activation(activation)(input_x_2)
        input_x_2 = multiply([input_x_2, input_x_2])
        power_feature = AveragePooling2D((1, length))(input_x_2) ## get average power of each channel each filter (batch, channel, 1, dim)
        power_feature = Reshape((input_channels, dim))(power_feature) ##(batch, channel, dim)
        x = Dense(input_channels, kernel_initializer= ki, kernel_regularizer = L)(power_feature)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dense(input_channels, kernel_initializer= ki, kernel_regularizer = L)(x)
        x = BatchNormalization()(x)
        x = Activation('softmax')(x)
        x_1 = Permute(dims=(2,1))(x)
        x_graph = add([x_1, x])/2
        input_x = Reshape((input_channels, -1))(input_x)
        x = tf.matmul(x_graph, input_x)
        x = Reshape((input_channels, length, dim))(x)
        x_graph = preprocess_adj(x_graph)


        return x, x_graph, power_feature
    
    return f




def irfanet(eeg_length, electrodes, kernLength, num_classes, fuse_type, e_value):
    eps = 1.1e-5

    EEG_input = Input(shape=(electrodes, eeg_length,1))
    x = Conv2D(filters=16, kernel_size= (1, kernLength), padding='same',use_bias=True, kernel_regularizer = L)(EEG_input) ##
    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)

    x, x_graph, power_feature = res_first(x,filters=[16,16],kernel_size=(1, kernLength),fuse_type = fuse_type)
    out = MultiGraphCNN(16, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_subsam(x,filters=[16,32],kernel_size=(1, kernLength),subsam=2, fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_nosub(x,filters=[16,32],kernel_size=(1, int(kernLength/2)), fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_subsam(x,filters=[16,32],kernel_size=(1, int(kernLength/2)),subsam=2, fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_nosub(x,filters=[32,32],kernel_size=(1, int(kernLength/4)), fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_subsam(x,filters=[32,32],kernel_size=(1, int(kernLength/4)),subsam=2, fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_nosub(x,filters=[32,32],kernel_size=(1, int(kernLength/8)), fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(32, 1, activation='elu')([power_feature, x_graph])
    x, x_graph, power_feature = res_subsam(x,filters=[32,32],kernel_size=(1, int(kernLength/8)),subsam=2, fuse_type = fuse_type)
    power_feature = concatenate([power_feature, out], axis=-1)
    out = MultiGraphCNN(16, 1, activation='elu')([power_feature, x_graph])
    out = tf.expand_dims(out, axis = -1)
    out = AveragePooling2D((1, 16))(out)
    out1 = Flatten()(out)
    out = Dense(num_classes,activation='softmax', name='graph_loss', kernel_regularizer = L)(out1)

    x = BatchNormalization(epsilon=eps, axis=-1)(x)
    x = Activation('elu')(x)

    x = DepthwiseConv2D((64, 1), use_bias = True, 
                                depth_multiplier = 2,
                                depthwise_constraint = max_norm(1.))(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(0.3)(x)

    x = SeparableConv2D(16, (1, 16),
                                    use_bias = True, padding = 'same')(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Dropout(0.3)(x)

    x = Flatten()(x)
    x1 = Dense(64, activation='elu', kernel_regularizer = L)(x)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x = Dense(num_classes,activation='softmax', name='convloss', kernel_regularizer = L)(x1)

    fused_x = concatenate([x1, out1], axis=-1)
    fused_x = Dense(num_classes,activation='softmax', name='fused_loss', kernel_regularizer = L)(fused_x)

    
        
    model = Model(EEG_input, outputs = [x,out,fused_x])
    
    l1 = mycrossentropy(fused_x, x, e_value)
    l2 = mycrossentropy(fused_x, out, e_value)

    
    model.add_loss((l1 + l2)/2)
    return model