#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, 
    GlobalAveragePooling2D, Concatenate, Dropout, 
    MaxPooling2D, Conv2DTranspose, Add
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
# LINKNET MODEL
# ========================

def residual_block(x, filters, stride=1, name_prefix=''):
    """Residual block for LinkNet encoder"""
    shortcut = x
    
    # First convolution
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False, 
               name=f'{name_prefix}_conv1')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Activation('relu', name=f'{name_prefix}_relu1')(x)
    
    # Second convolution
    x = Conv2D(filters, (3, 3), padding='same', use_bias=False, 
               name=f'{name_prefix}_conv2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
    
    # Adjust shortcut if dimensions don't match
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', use_bias=False,
                         name=f'{name_prefix}_shortcut_conv')(shortcut)
        shortcut = BatchNormalization(name=f'{name_prefix}_shortcut_bn')(shortcut)
    
    # Add shortcut and apply activation
    x = Add(name=f'{name_prefix}_add')([x, shortcut])
    x = Activation('relu', name=f'{name_prefix}_relu2')(x)
    
    return x

def decoder_block(encoder_output, skip_connection, filters, name_prefix=''):
    """Decoder block for LinkNet"""
    # First convolution to reduce channels
    x = Conv2D(filters // 4, (1, 1), use_bias=False, name=f'{name_prefix}_conv1')(encoder_output)
    x = BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = Activation('relu', name=f'{name_prefix}_relu1')(x)
    
    # Transpose convolution for upsampling
    x = Conv2DTranspose(filters // 4, (3, 3), strides=2, padding='same', use_bias=False,
                       name=f'{name_prefix}_deconv')(x)
    x = BatchNormalization(name=f'{name_prefix}_deconv_bn')(x)
    x = Activation('relu', name=f'{name_prefix}_deconv_relu')(x)
    
    # Final convolution
    x = Conv2D(filters, (1, 1), use_bias=False, name=f'{name_prefix}_conv2')(x)
    x = BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = Activation('relu', name=f'{name_prefix}_relu2')(x)
    
    # Add skip connection
    x = Add(name=f'{name_prefix}_add')([x, skip_connection])
    
    return x

def linknet_model(input_size=(512, 512, 3)):
    """LinkNet implementation"""
    inputs = Input(input_size, name='input')

    # Initial convolution
    x = Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False, name='initial_conv')(inputs)
    x = BatchNormalization(name='initial_bn')(x)
    x = Activation('relu', name='initial_relu')(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same', name='initial_pool')(x)

    # Encoder blocks
    e1 = residual_block(x, 64, name_prefix='e1_block1')
    e1 = residual_block(e1, 64, name_prefix='e1_block2')

    e2 = residual_block(e1, 128, stride=2, name_prefix='e2_block1')
    e2 = residual_block(e2, 128, name_prefix='e2_block2')

    e3 = residual_block(e2, 256, stride=2, name_prefix='e3_block1')
    e3 = residual_block(e3, 256, name_prefix='e3_block2')

    e4 = residual_block(e3, 512, stride=2, name_prefix='e4_block1')
    e4 = residual_block(e4, 512, name_prefix='e4_block2')

    # Decoder blocks
    d4 = decoder_block(e4, e3, 256, name_prefix='d4')
    d3 = decoder_block(d4, e2, 128, name_prefix='d3')
    d2 = decoder_block(d3, e1, 64, name_prefix='d2')

    # Final upsampling
    d1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same', name='final_deconv1')(d2)
    d1 = BatchNormalization(name='final_bn1')(d1)
    d1 = Activation('relu', name='final_relu1')(d1)

    d1 = Conv2D(32, (3, 3), padding='same', name='final_conv1')(d1)
    d1 = BatchNormalization(name='final_bn2')(d1)
    d1 = Activation('relu', name='final_relu2')(d1)

    # Final upsampling to original size
    final = Conv2DTranspose(32, (2, 2), strides=2, padding='same', name='final_deconv2')(d1)
    final = BatchNormalization(name='final_bn3')(final)
    final = Activation('relu', name='final_relu3')(final)

    outputs = Conv2D(1, (1, 1), activation='sigmoid', name='output')(final)

    model = Model(inputs=[inputs], outputs=[outputs], name='LinkNet')
    return model

# ========================
# DATA GENERATOR
# ========================

class LinkNetDataGenerator(Sequence):
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

def train_linknet(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir, 
                  epochs=50, batch_size=4, use_focal_loss=True):
    """Train LinkNet model"""
    
    # Create data generators
    train_gen = LinkNetDataGenerator(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        batch_size=batch_size,
        shuffle=True
    )

    val_gen = LinkNetDataGenerator(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        batch_size=batch_size,
        shuffle=False
    )

    # Create model
    print("Creating LinkNet model...")
    model = linknet_model()
    
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
            "best_linknet_model.keras",
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
        tf.keras.callbacks.CSVLogger('linknet_training_log.csv')
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
    model, history = train_linknet(
        train_image_dir=train_image_dir,
        train_mask_dir=train_mask_dir,
        val_image_dir=val_image_dir,
        val_mask_dir=val_mask_dir,
        epochs=2,
        batch_size=4,
        use_focal_loss=True  # Good for line segmentation
    )
    
    # Save final model
    model.save("final_linknet_model.keras")
    print("Training completed! Model saved as 'final_linknet_model.keras'")
    
    # Print final training results
    print("\nFinal Training Results:")
    print(f"Best Validation Dice: {max(history.history['val_dice_coefficient']):.4f}")
    print(f"Best Validation IoU: {max(history.history['val_iou_score']):.4f}")
    print(f"Best Validation F1: {max(history.history['val_f1_score']):.4f}")


# In[ ]:




