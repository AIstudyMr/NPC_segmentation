from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, Concatenate, Input
from tensorflow.keras.models import Model

def residual_block_1(inputs, num_filters, strides=1):
        x = Conv2D(num_filters, 3, padding="same", strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
        s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
        x = x + s
        return x

def residual_block_2(inputs, num_filters, strides=1):
    x = BatchNormalization()(inputs)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same", strides=strides)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(num_filters, 3, padding="same", strides=1)(x)
    s = Conv2D(num_filters, 1, padding="same", strides=strides)(inputs)
    x1 = BatchNormalization()(s)
    x2 = Activation("relu")(x1)
    x = x + x2
    return x

def decoder_block(inputs, skip_features, num_filters):
    x = UpSampling2D((2, 2))(inputs)
    x = Concatenate()([x, skip_features])
    x = residual_block_2(x, num_filters, strides=1)
    return x

def conv2d_bn(x,filters,num_row,num_col,padding='same',stride=1,dilation_rate=1,relu=True):
    x = Conv2D(
        filters, (num_row, num_col),
        strides=(stride,stride),
        padding=padding,
        dilation_rate=(dilation_rate, dilation_rate),
        use_bias=False)(x)
    x = BatchNormalization(scale=False)(x)
    if relu:    
        x = Activation("relu")(x)
    return x

def BasicRFB(x,input_filters,output_filters,stride=1):
        input_filters_div = input_filters//8

        branch_0 = conv2d_bn(x,input_filters_div*2,1,1,stride=stride)
        branch_0 = conv2d_bn(branch_0,input_filters_div*2,3,3,relu=False)

        branch_1 = conv2d_bn(x,input_filters_div,1,1)
        branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,stride=stride)
        branch_1 = conv2d_bn(branch_1,input_filters_div*2,3,3,dilation_rate=3,relu=False)
        
        branch_2 = conv2d_bn(x,input_filters_div,1,1)
        branch_2 = conv2d_bn(branch_2,(input_filters_div//2)*3,3,3)
        branch_2 = conv2d_bn(branch_2,input_filters_div*2,3,3,stride=stride)
        branch_2 = conv2d_bn(branch_2,input_filters_div*2,3,3,dilation_rate=5,relu=False)

        out = concatenate([branch_0,branch_1,branch_2],axis=-1)
        out = conv2d_bn(out,output_filters,1,1,relu=False)

        short = conv2d_bn(x,output_filters,1,1,stride=stride,relu=False)
        out = Add()([out, short])
        out = Activation("relu")(out)
        return out

def BasicRFB_A(x, input_filters, output_filters, stride=1):
    input_filters_div = input_filters // 8
    branch_0 = conv2d_bn(x, input_filters_div, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div, 3, 3, relu=False)
    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 2, 2)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 2, 2)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 3, 3, dilation_rate=3, relu=False)
    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 1, 5)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 5, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 3, 3, dilation_rate=5, relu=False)
    out = concatenate([branch_0, branch_1, branch_2], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)
    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out

def BasicRFB_B(x, input_filters, output_filters, stride=1):
    input_filters_div = input_filters // 8

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div * 2, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 1, 3, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 1, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, dilation_rate=5, relu=False)

    out = concatenate([branch_0, branch_1, branch_2], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


def BasicRFB_C(x, input_filters, output_filters, stride=1):
    input_filters_div = input_filters // 8

    branch_0 = conv2d_bn(x, input_filters_div * 2, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div * 2, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 1, 3, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 1, stride=stride)
    branch_1 = conv2d_bn(branch_1, input_filters_div * 2, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 1, 5, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 5, 1, stride=stride)
    branch_2 = conv2d_bn(branch_2, input_filters_div * 2, 3, 3, dilation_rate=5, relu=False)

    out = concatenate([branch_0, branch_1, branch_2], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


def BasicRFB_D(x, input_filters, output_filters, stride=1):
    input_filters_div = input_filters // 8

    branch_0 = conv2d_bn(x, input_filters_div, 1, 1, stride=stride)
    branch_0 = conv2d_bn(branch_0, input_filters_div, 3, 3, relu=False)

    branch_1 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 2, 2)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 2, 2)
    branch_1 = conv2d_bn(branch_1, input_filters_div, 3, 3, dilation_rate=3, relu=False)

    branch_2 = conv2d_bn(x, input_filters_div, 1, 1)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 3, 3)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 3, 3)
    branch_2 = conv2d_bn(branch_2, input_filters_div, 3, 3, dilation_rate=5, relu=False)

    out = concatenate([branch_0, branch_1, branch_2], axis=-1)
    out = conv2d_bn(out, output_filters, 1, 1, relu=False)

    short = conv2d_bn(x, output_filters, 1, 1, stride=stride, relu=False)
    out = Lambda(lambda x: x[0] + x[1])([out, short])
    out = Activation("relu")(out)
    return out


# build  model
inputs = Input((im_height, im_width, depth))          
# Endocer
s1 = residual_block_1(inputs, 64, strides=1)       
s2 = residual_block_2(s1, 128, strides=2)    
s3 = residual_block_2(s2, 256, strides=2)   
# brige module
s4= residual_block_1(s3, 512, strides=2)    
Ra=BasicRFB_A(s4,512,256,stride=1)            
b1= Conv2D(256, 3, padding="same", strides=2)(Ra)
# Dedocer
d1 = decoder_block(Ra, s3, 256)                 
d2 = decoder_block(d1, s2, 128)                
d3 = decoder_block(d2, s1, 64)                

outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d3)
model = Model(inputs, outputs)

