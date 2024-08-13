#The GANs model with Wasserstein distance along with helper functions

#-*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers

class LeakyReLU(layers.Layer):
    def __init__(self, leak=0.2, **kwargs):
        super(LeakyReLU, self).__init__(**kwargs)
        self.leak = leak

    def call(self, X):
        return tf.maximum(X, self.leak * X)

class GAN():
    def __init__(
            self,
            batch_size=32,
            image_shape=[24,24,1],
            dim_z=100,
            dim_y=6,
            dim_W1=1024,
            dim_W2=128,
            dim_W3=64,
            dim_channel=1,
            ):

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = layers.Dense(dim_W1, use_bias=False, name='gen_W1')
        self.gen_W2 = layers.Dense(dim_W2 * 6 * 6, use_bias=False, name='gen_W2')
        self.gen_conv2d_transpose1 = layers.Conv2DTranspose(filters=dim_W3, kernel_size=5, strides=2, padding='same', use_bias=False, name='gen_conv2d_transpose1')
        self.gen_conv2d_transpose2 = layers.Conv2DTranspose(filters=dim_channel, kernel_size=5, strides=2, padding='same', use_bias=False, name='gen_conv2d_transpose2')

        self.discrim_conv2d_1 = layers.Conv2D(filters=dim_W3, kernel_size=5, strides=2, padding='same', use_bias=False, name='discrim_conv2d_1')
        self.discrim_conv2d_2 = layers.Conv2D(filters=dim_W2, kernel_size=5, strides=2, padding='same', use_bias=False, name='discrim_conv2d_2')
        self.discrim_W3 = layers.Dense(dim_W1, use_bias=False, name='discrim_W3')
        self.discrim_W4 = layers.Dense(1, use_bias=False, name='discrim_W4')
        self.leaky_relu = LeakyReLU()

    def batchnormalize(self, X, epsilon=1e-8):
        """Aplica normalización por lotes"""
        return layers.BatchNormalization()(X)

    def build_model(self):
      Z = tf.keras.Input(name='Z', shape=(self.dim_z,), dtype=tf.dtypes.float32)
      Y = tf.keras.Input(name='Y', shape=(self.dim_y,), dtype=tf.dtypes.float32)
      image_real = tf.keras.Input(name='image_real', shape=self.image_shape, dtype=tf.dtypes.float32)

      h4 = self.generate(Z, Y)
      image_gen = layers.Activation('sigmoid')(h4)  # Cambiado para usar la capa Activation de Keras

      raw_real2 = self.discriminate(image_real, Y)
      p_real = tf.reduce_mean(raw_real2)

      raw_gen2 = self.discriminate(image_gen, Y)
      p_gen = tf.reduce_mean(raw_gen2)

      discrim_cost = tf.reduce_sum(raw_real2) - tf.reduce_sum(raw_gen2)
      gen_cost = -tf.reduce_mean(raw_gen2)
      return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        yb = layers.Reshape([1, 1, self.dim_y])(Y)
        X = layers.Concatenate(axis=3)([image, yb * tf.ones([self.batch_size, 24, 24, self.dim_y])])
        h1 = self.leaky_relu(self.discrim_conv2d_1(X))
        h1 = layers.Concatenate(axis=3)([h1, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])])
        h2 = self.leaky_relu(self.batchnormalize(self.discrim_conv2d_2(h1)))
        h2 = layers.Reshape([self.batch_size, -1])(h2)  # Flatten the tensor
        h2_shape = tf.shape(h2)
        h2 = layers.Reshape([h2_shape[1]])(h2)  # Reshape to 2D tensor
        h2 = layers.Concatenate(axis=1)([h2, Y])
        h3 = self.leaky_relu(self.batchnormalize(self.discrim_W3(h2)))
        return h3

    def generate(self, Z, Y):
        Z = layers.Concatenate(axis=1)([Z, Y])
        h1 = layers.ReLU()(self.batchnormalize(self.gen_W1(Z)))
        h1 = layers.Concatenate(axis=1)([h1, Y])
        h2 = layers.ReLU()(self.batchnormalize(self.gen_W2(h1)))
        h2 = layers.Reshape([6, 6, self.dim_W2])(h2)  # Cambiado para usar Reshape de Keras
        yb = layers.Reshape([1, 1, self.dim_y])(Y)
        h2 = layers.Concatenate(axis=3)([h2, yb * tf.ones([self.batch_size, 6, 6, self.dim_y])])

        h3 = layers.ReLU()(self.batchnormalize(self.gen_conv2d_transpose1(h2)))
        h3 = layers.Concatenate(axis=3)([h3, yb * tf.ones([self.batch_size, 12, 12, self.dim_y])])

        h4 = self.gen_conv2d_transpose2(h3)
        return h4

# Función lrelu como estaba definida en el código original (asumiendo que ya existe)
def lrelu(X, leak=0.2):
    return tf.maximum(X, leak * X)


