
"""
@title: The Keras Implementation of Residual Attention Network
@标题：注意力残差网络在Keras框架下的复现

@author: Abel·SunMe
@作者：松明

@date: 2nd January, 2019
    
"""

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Activation, BatchNormalization,\
     MaxPooling2D, UpSampling2D, AveragePooling2D, Add, Multiply

def ResidualUnit(x,paras,stride=1):

    x_ = BatchNormalization()(x)
    x_ = Activation('relu')(x_)
    x_ = Conv2D(filters=paras[0][2], kernel_size=(paras[0][:2]),
                padding='same', strides=stride)(x_)

    x_ = BatchNormalization()(x_)
    x_ = Activation('relu')(x_)
    x_ = Conv2D(filters=paras[1][2], kernel_size=(paras[1][:2]),
                padding='same')(x_)

    x_ = BatchNormalization()(x_)
    x_ = Activation('relu')(x_)
    x_ = Conv2D(filters=paras[2][2], kernel_size=(paras[2][:2]),
                padding='same')(x_)
    
    if(int(x.shape[-1])!=paras[2][2] or stride!=1):
        x_ = Add()([x_,Conv2D(filters=paras[2][2], kernel_size=(1,1),
                              padding='same', strides=stride)(x)])
    else:
        x_ = Add()([x_,x])

    return x_

#Shape Definition of InputData
x0 = Input(shape=(224,224,3,), name='Input_Image')

#Input Layers
x1 = Conv2D(filters=64, kernel_size=(7,7), strides=2, padding='same')(x0)
x1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x1)

#Attention Module stage 1
x2 = ResidualUnit(x1, paras=([1,1,64],[3,3,64],[1,1,256]))

#stage 1 Trunk Branch
x3 = ResidualUnit(x2, paras=([1,1,64],[3,3,64],[1,1,256]))
x3 = ResidualUnit(x3, paras=([1,1,64],[3,3,64],[1,1,256]))

#stage 1 Soft Mask Branch
xs1_a1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x2)
xs1_a1 = ResidualUnit(xs1_a1, paras=([1,1,64],[3,3,64],[1,1,256]))

xs1_a2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(xs1_a1)
xs1_a2 = ResidualUnit(xs1_a2, paras=([1,1,64],[3,3,64],[1,1,256]))

xs1_a3 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(xs1_a2)
xs1_a3 = ResidualUnit(xs1_a3, paras=([1,1,64],[3,3,64],[1,1,256]))

xs1_r3 = ResidualUnit(xs1_a3, paras=([1,1,64],[3,3,64],[1,1,256]))
xs1_r3 = UpSampling2D(interpolation='bilinear')(xs1_r3)
xs1_r3 = Add()([xs1_r3,xs1_a2])

xs1_r2 = ResidualUnit(xs1_r3, paras=([1,1,64],[3,3,64],[1,1,256]))
xs1_r2 = UpSampling2D(interpolation='bilinear')(xs1_r2)
xs1_r2 = Add()([xs1_r2,xs1_a1])

xs1_r1 = ResidualUnit(xs1_r2, paras=([1,1,64],[3,3,64],[1,1,256]))
xs1_r1 = UpSampling2D(interpolation='bilinear')(xs1_r1)
xs1_r1 = Conv2D(filters=64, kernel_size=(1,1), padding='same')(xs1_r1)
xs1_r1 = Conv2D(filters=256, kernel_size=(1,1), padding='same')(xs1_r1)
xs1_r1 = Activation('sigmoid')(xs1_r1)

#stage 1 Branches Mixing
x4 = Multiply()([x3,xs1_r1])
x4 = Add()([x3,x4])
x4 = ResidualUnit(x4, paras=([1,1,64],[3,3,64],[1,1,256]))

#stage 1 DownSampling ResidualUnit
x5 = ResidualUnit(x4, paras=([1,1,128],[3,3,128],[1,1,512]), stride=2)


#Attention Module stage 2
x6 = ResidualUnit(x5, paras=([1,1,128],[3,3,128],[1,1,512]))

#stage 2 Trunk Branch
x7 = ResidualUnit(x6, paras=([1,1,128],[3,3,128],[1,1,512]))
x7 = ResidualUnit(x7, paras=([1,1,128],[3,3,128],[1,1,512]))

#stage 2 Soft Mask Branch
xs2_a1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x6)
xs2_a1 = ResidualUnit(xs2_a1, paras=([1,1,128],[3,3,128],[1,1,512]))

xs2_a2 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(xs2_a1)
xs2_a2 = ResidualUnit(xs2_a2, paras=([1,1,128],[3,3,128],[1,1,512]))

xs2_r2 = ResidualUnit(xs2_a2, paras=([1,1,128],[3,3,128],[1,1,512]))
xs2_r2 = UpSampling2D(interpolation='bilinear')(xs2_r2)
xs2_r2 = Add()([xs2_r2,xs2_a1])

xs2_r1 = ResidualUnit(xs2_r2, paras=([1,1,128],[3,3,128],[1,1,512]))
xs2_r1 = UpSampling2D(interpolation='bilinear')(xs2_r1)
xs2_r1 = Conv2D(filters=128, kernel_size=(1,1), padding='same')(xs2_r1)
xs2_r1 = Conv2D(filters=512, kernel_size=(1,1), padding='same')(xs2_r1)
xs2_r1 = Activation('sigmoid')(xs2_r1)

#stage 2 Branches Mixing
x8 = Multiply()([x7,xs2_r1])
x8 = Add()([x7,x8])
x8 = ResidualUnit(x8, paras=([1,1,128],[3,3,128],[1,1,512]))

#stage 2 DownSampling ResidualUnit
x9 = ResidualUnit(x8, paras=([1,1,256],[3,3,256],[1,1,1024]), stride=2)


#Attention Module stage 3
x10 = ResidualUnit(x9, paras=([1,1,256],[3,3,256],[1,1,1024]))

#stage 3 Trunk Branch
x11 = ResidualUnit(x10, paras=([1,1,256],[3,3,256],[1,1,1024]))
x11 = ResidualUnit(x11, paras=([1,1,256],[3,3,256],[1,1,1024]))

#stage 3 Soft Mask Branch
xs3_a1 = MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x10)
xs3_a1 = ResidualUnit(xs3_a1, paras=([1,1,256],[3,3,256],[1,1,1024]))

xs3_r1 = ResidualUnit(xs3_a1, paras=([1,1,256],[3,3,256],[1,1,1024]))
xs3_r1 = UpSampling2D(interpolation='bilinear')(xs3_r1)
xs3_r1 = Conv2D(filters=256, kernel_size=(1,1), padding='same')(xs3_r1)
xs3_r1 = Conv2D(filters=1024, kernel_size=(1,1), padding='same')(xs3_r1)
xs3_r1 = Activation('sigmoid')(xs3_r1)

#stage 3 Branches Mixing
x12 = Multiply()([x11,xs3_r1])
x12 = Add()([x11,x12])
x12 = ResidualUnit(x12, paras=([1,1,256],[3,3,256],[1,1,1024]))

#stage 3 DownSampling ResidualUnit
x13 = ResidualUnit(x12, paras=([1,1,512],[3,3,512],[1,1,2048]), stride=2)
x13 = ResidualUnit(x13, paras=([1,1,512],[3,3,512],[1,1,2048]))
x13 = ResidualUnit(x13, paras=([1,1,512],[3,3,512],[1,1,2048]))

#Output Layers
x14 = AveragePooling2D(pool_size=(7,7),strides=1)(x13)
x14 = Dense(1000)(x14)
x14 = Activation('softmax')(x14)

#Compile Model
model = Model(inputs=x0, outputs=x14)
model.compile

"""
You can Train this Model by parsing
    
model.fit(Data_of_Images,Labels)

as the input images were already resized
to 224x224 resolution.
    
"""

from keras.utils import plot_model
plot_model(model, to_file='model.png')
