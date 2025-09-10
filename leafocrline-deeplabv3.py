#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    GlobalAveragePooling2D, Concatenate, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from tensorflow.keras import layers

# ========================
# LOSS FUNCTIONS AND METRICS
# ========================

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """Dice coefficient metric"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss function"""
    return 1 - dice_coefficient(y_true, y_pred)

def binary_crossentropy_dice_loss(y_true, y_pred):
    """Combined binary crossentropy and dice loss"""
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def focal_loss(y_true, y_pred, alpha=0.8, gamma=2.0):
    """Focal loss for handling class imbalance"""
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred)
    focal_loss_val = -alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    
    return K.mean(focal_loss_val)

def combined_focal_dice_loss(y_true, y_pred):
    """Combined focal and dice loss"""
    focal = focal_loss(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return focal + dice

def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU (Jaccard) coefficient"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

def precision(y_true, y_pred):
    """Precision metric"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    """Recall metric"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    """F1 score metric"""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# ========================
# DEEPLABV3+ MODEL
# ========================

def depthwise_separable_conv(x, filters, kernel_size=3, strides=1, dilation_rate=1, name_prefix=''):
    """Depthwise separable convolution"""
    x = layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding='same',
        use_bias=False,
        name=f'{name_prefix}_depthwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_depthwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_depthwise_relu')(x)

    x = Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=f'{name_prefix}_pointwise'
    )(x)
    x = BatchNormalization(name=f'{name_prefix}_pointwise_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_pointwise_relu')(x)

    return x

def atrous_spatial_pyramid_pooling(x, output_stride=16):
    """Atrous Spatial Pyramid Pooling (ASPP) module"""
    # 1x1 convolution
    b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
    b0 = BatchNormalization(name='aspp0_bn')(b0)
    b0 = Activation('relu', name='aspp0_activation')(b0)

    # Atrous convolutions with different rates
    atrous_rates = (6, 12, 18) if output_stride == 16 else (12, 24, 36)

    b1 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[0], 
                use_bias=False, name='aspp1')(x)
    b1 = BatchNormalization(name='aspp1_bn')(b1)
    b1 = Activation('relu', name='aspp1_activation')(b1)

    b2 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[1], 
                use_bias=False, name='aspp2')(x)
    b2 = BatchNormalization(name='aspp2_bn')(b2)
    b2 = Activation('relu', name='aspp2_activation')(b2)

    b3 = Conv2D(256, (3, 3), padding='same', dilation_rate=atrous_rates[2], 
                use_bias=False, name='aspp3')(x)
    b3 = BatchNormalization(name='aspp3_bn')(b3)
    b3 = Activation('relu', name='aspp3_activation')(b3)

    # Global average pooling branch
    b4 = GlobalAveragePooling2D()(x)
    b4 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(b4)
    b4 = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, 1))(b4)
    b4 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='image_pooling')(b4)
    b4 = BatchNormalization(name='image_pooling_bn')(b4)
    b4 = Activation('relu', name='image_pooling_activation')(b4)

    # Resize to match feature map dimensions
    def resize_to_feature_map(inputs):
        feature_map, pooled_features = inputs
        shape = tf.shape(feature_map)
        return tf.image.resize(pooled_features, [shape[1], shape[2]])

    b4 = tf.keras.layers.Lambda(resize_to_feature_map)([x, b4])

    # Concatenate all branches
    x = Concatenate()([b0, b1, b2, b3, b4])

    # Final projection
    x = Conv2D(256, (1, 1), padding='same', use_bias=False, name='concat_projection')(x)
    x = BatchNormalization(name='concat_projection_bn')(x)
    x = Activation('relu', name='concat_projection_activation')(x)
    x = Dropout(0.1)(x)

    return x

def deeplabv3plus_model(input_size=(512, 512, 3), output_stride=16):
    """DeepLabv3+ model for binary segmentation"""
    inputs = Input(input_size)

    # ENCODER (Entry flow)
    x = Conv2D(32, (3, 3), strides=2, padding='same', use_bias=False, name='entry_conv1')(inputs)
    x = BatchNormalization(name='entry_conv1_bn')(x)
    x = Activation('relu', name='entry_conv1_relu')(x)

    x = Conv2D(64, (3, 3), padding='same', use_bias=False, name='entry_conv2')(x)
    x = BatchNormalization(name='entry_conv2_bn')(x)
    x = Activation('relu', name='entry_conv2_relu')(x)

    # Block 1
    residual = Conv2D(128, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = depthwise_separable_conv(x, 128, name_prefix='block1_1')
    x = depthwise_separable_conv(x, 128, name_prefix='block1_2')
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Block 2
    residual = Conv2D(256, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 256, name_prefix='block2_1')
    x = depthwise_separable_conv(x, 256, name_prefix='block2_2')
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Save low-level features for decoder
    low_level_features = x

    # Block 3
    residual = Conv2D(728, (1, 1), strides=2, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='block3_1')
    x = depthwise_separable_conv(x, 728, name_prefix='block3_2')
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)
    x = layers.Add()([x, residual])

    # Middle flow (8 repeated blocks)
    for i in range(8):
        residual = x
        x = Activation('relu')(x)
        x = depthwise_separable_conv(x, 728, name_prefix=f'middle_{i}_1')
        x = depthwise_separable_conv(x, 728, name_prefix=f'middle_{i}_2')
        x = depthwise_separable_conv(x, 728, name_prefix=f'middle_{i}_3')
        x = layers.Add()([x, residual])

    # Exit flow
    residual = Conv2D(1024, (1, 1), strides=1, padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu')(x)
    x = depthwise_separable_conv(x, 728, name_prefix='exit_1')
    x = depthwise_separable_conv(x, 1024, name_prefix='exit_2')
    x = layers.Add()([x, residual])

    x = depthwise_separable_conv(x, 1536, name_prefix='exit_3')
    x = depthwise_separable_conv(x, 1536, name_prefix='exit_4')
    x = depthwise_separable_conv(x, 2048, name_prefix='exit_5')

    # ASPP
    x = atrous_spatial_pyramid_pooling(x, output_stride)

    # DECODER
    # Process low-level features
    low_level_features = Conv2D(48, (1, 1), padding='same', use_bias=False, 
                               name='low_level_projection')(low_level_features)
    low_level_features = BatchNormalization(name='low_level_projection_bn')(low_level_features)
    low_level_features = Activation('relu', name='low_level_projection_relu')(low_level_features)

    # Upsample encoder features to match low-level features
    def upsample_to_match(inputs):
        encoder_features, low_level_features = inputs
        shape = tf.shape(low_level_features)
        return tf.image.resize(encoder_features, [shape[1], shape[2]])

    x = tf.keras.layers.Lambda(upsample_to_match)([x, low_level_features])

    # Concatenate with low-level features
    x = Concatenate()([x, low_level_features])

    # Decoder convolutions
    x = depthwise_separable_conv(x, 256, name_prefix='decoder_1')
    x = depthwise_separable_conv(x, 256, name_prefix='decoder_2')

    # Final upsampling to input resolution
    x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, [512, 512]))(x)

    # Output layer
    outputs = Conv2D(1, kernel_size=1, activation='sigmoid', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ========================
# DATA GENERATOR
# ========================

class DeepLabDataGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=4, img_size=(512, 512), 
                 shuffle=True, binary_threshold=127):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        self.binary_threshold = binary_threshold

        # Get image files
        self.file_names = sorted([f for f in os.listdir(image_dir)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"Found {len(self.file_names)} images in {image_dir}")
        
        if self.shuffle:
            np.random.shuffle(self.file_names)

    def __len__(self):
        return len(self.file_names) // self.batch_size

    def __getitem__(self, index):
        batch_files = self.file_names[index * self.batch_size:(index + 1) * self.batch_size]
        X, Y = [], []

        for fname in batch_files:
            img_path = os.path.join(self.image_dir, fname)
            
            # Find corresponding mask
            base_name = os.path.splitext(fname)[0]
            mask_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                potential_mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(potential_mask_path):
                    mask_path = potential_mask_path
                    break

            if mask_path is None:
                continue

            # Load and preprocess image
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.img_size)
            img = img.astype(np.float32) / 255.0

            # Load and preprocess mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.img_size)
            mask = (mask > self.binary_threshold).astype(np.float32)
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.file_names)

# ========================
# TRAINING FUNCTION
# ========================

def train_deeplabv3plus(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, 
                       epochs=50, batch_size=4, use_focal_loss=True):
    """Train DeepLabv3+ model"""
    
    # Create data generators
    train_gen = DeepLabDataGenerator(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = DeepLabDataGenerator(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    print("Creating DeepLabv3+ model...")
    model = deeplabv3plus_model()
    
    # Print model info
    print(f"Model parameters: {model.count_params():,}")
    
    # Choose loss function
    loss_function = combined_focal_dice_loss if use_focal_loss else binary_crossentropy_dice_loss
    loss_name = "Focal+Dice" if use_focal_loss else "BCE+Dice"
    print(f"Using {loss_name} loss function")
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=loss_function,
        metrics=[dice_coefficient, iou_score, precision, recall, f1_score]
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "test_best_deeplabv3plus_model.keras",
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient',
            mode='max',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger('deeplabv3plus_training_log.csv')
    ]
    
    # Train model
    print(f"Starting training for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return model, history

# ========================
# PREDICTION FUNCTION
# ========================

def predict_image(model_path, image_path, target_size=(512, 512)):
    """Make prediction on a single image"""
    # Load model
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'dice_loss': dice_loss,
        'binary_crossentropy_dice_loss': binary_crossentropy_dice_loss,
        'combined_focal_dice_loss': combined_focal_dice_loss,
        'iou_score': iou_score,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
    
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    
    img_resized = cv2.resize(img, target_size)
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Predict
    pred = model.predict(img_batch)[0]
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
    # Resize back to original size
    pred_resized = cv2.resize(pred_binary.squeeze(), (original_size[1], original_size[0]))
    
    return pred_resized

# ========================
# MAIN TRAINING SCRIPT
# ========================

if __name__ == "__main__":
    # Set your data directories
    train_image_dir = r"D:\phd\OneDrive\AMRITA\line segmentaion\for labelling\qualitywise implementaion\dataset_split\qualitywisetogethjer\image_patches\test_patches_512"
    train_mask_dir = r"D:\phd\OneDrive\AMRITA\line segmentaion\for labelling\qualitywise implementaion\dataset_split\qualitywisetogethjer\maks_patches\test_patches_512"
    val_image_dir = r"D:\phd\OneDrive\AMRITA\line segmentaion\for labelling\qualitywise implementaion\dataset_split\qualitywisetogethjer\image_patches\val_patches_512"
    val_mask_dir = r"D:\phd\OneDrive\AMRITA\line segmentaion\for labelling\qualitywise implementaion\dataset_split\qualitywisetogethjer\maks_patches\val_patches_512"
    
    # Train the model
    model, history = train_deeplabv3plus(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        epochs=2,
        batch_size=4,
        use_focal_loss=True  # Good for line segmentation
    )
    
    # Save final model
    model.save("final_deeplabv3plus_model.keras")
    print("Training completed! Model saved as 'tesitn_final_deeplabv3plus_model.keras'")
    
    # Print final training results
    print("\nFinal Training Results:")
    print(f"Best Validation Dice: {max(history.history['val_dice_coefficient']):.4f}")
    print(f"Best Validation IoU: {max(history.history['val_iou_score']):.4f}")
    print(f"Best Validation F1: {max(history.history['val_f1_score']):.4f}")


# In[ ]:




