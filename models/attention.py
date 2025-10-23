### bloc cbam

import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Conv2D

class CBAM(Layer):
    """
    CBAM = Channel Attention + spatial attention

    value : ratio 8 et kernel_size = 7 taille standart vue ds le cours
    """

    def __init__(self, ratio=8, kernel_size=7, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size= kernel_size

    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.ratio, 1)

        # channel Attention
        self.gap = GlobalAveragePooling2D()
        self.gmp = GlobalMaxPooling2D()
        self.fc1 = Dense(hidden, activation='relu', kernel_initializer="he_normal", use_bias=True)
        self.fc2 = Dense(channels, activation=None, kernel_initializer="he_normal", use_bias=True)

        # spacial attention aide à la symétrie
        self.spatial_conv =  Conv2D(1, self.kernel_size, padding='same',
                                   activation='sigmoid', kernel_initializer='he_normal')
        
        super(CBAM, self).build(input_shape)


    def call(self, inputs):
        # Channel Attention
        avg_pool = self.fc2(self.fc1(self.gap(inputs)))
        max_pool = self.fc2(self.fc1(self.gmp(inputs)))
        channel_attn = tf.nn.sigmoid(avg_pool + max_pool)
        channel_attn = tf.reshape(channel_attn, (-1, 1, 1, tf.shape(inputs)[-1]))
        
        # channel attention
        features = inputs * channel_attn  # x en features
        
        # Spatial Attention
        avg_spatial = tf.reduce_mean(features, axis=-1, keepdims=True)  # features au lieu de x
        max_spatial = tf.reduce_max(features, axis=-1, keepdims=True)   # features au lieu de x
        spatial = tf.concat([avg_spatial, max_spatial], axis=-1)
        spatial_attn = self.spatial_conv(spatial)
    
        return features * spatial_attn  # features au lieu de x
    
    def get_config(self):
        config = super(CBAM, self).get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config
    

# amelioration  avec Squeeze-and-Excitation (SE) Block
class SEBlock(Layer):
    """
    SE Block Alternative moderne à CBAM 2018
    + léger performances similaires
    
    Valeur : ratio=16  compression standard SE + agressif que CBAM
    """
    
    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        channels = int(input_shape[-1])
        hidden = max(channels // self.ratio, 1)
        
        self.gap = GlobalAveragePooling2D()
        self.fc1 = Dense(hidden, activation='relu', kernel_initializer='he_normal')
        self.fc2 = Dense(channels, activation='sigmoid', kernel_initializer='he_normal')
        
        super(SEBlock, self).build(input_shape)
    
    def call(self, inputs):
        scale = self.fc2(self.fc1(self.gap(inputs)))
        scale = tf.reshape(scale, (-1, 1, 1, tf.shape(inputs)[-1]))
        return inputs * scale
    
    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"ratio": self.ratio})
        return config