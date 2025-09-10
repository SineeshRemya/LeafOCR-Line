#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def fcn16_model(input_size=(512, 512, 3)):
    """FCN-16s implementation for binary segmentation"""
    inputs = Input(input_size)

    # VGG-like encoder
    # Block 1
    x = conv_block(inputs, 64)
    x = conv_block(x, 64)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = conv_block(pool1, 128)
    x = conv_block(x, 128)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = conv_block(pool2, 256)
    x = conv_block(x, 256)
    x = conv_block(x, 256)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = conv_block(pool3, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = conv_block(pool4, 512)
    x = conv_block(x, 512)
    x = conv_block(x, 512)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Classifier (replace FC layers with convolutions)
    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(pool5)
    x = Dropout(0.5)(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)

    # Score layers
    score_fr = Conv2D(1, (1, 1), padding='same')(x)  # Final score
    score_pool4 = Conv2D(1, (1, 1), padding='same')(pool4)  # Score from pool4

    # Upsample score_fr by 2
    upscore2 = Conv2DTranspose(1, (4, 4), strides=(2, 2), padding='same')(score_fr)

    # Fuse with pool4
    fuse_pool4 = add([upscore2, score_pool4])

    # Final upsampling by 16
    outputs = Conv2DTranspose(1, (32, 32), strides=(16, 16), padding='same', activation='sigmoid')(fuse_pool4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

