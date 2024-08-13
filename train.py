#The main training function
#For event-based scenario generation and spatial scenario generation, implement the code with labels or 
#reshape the imput samples to spatio-temporal samples respectively.
#16 is the maximum value for wind capacity we use. Change to the customized max value for normalized data

import tensorflow as tf
from model import GAN  # Asumiendo que la clase GAN está en un archivo llamado model.py

# Definir las dimensiones
batch_size = 32
image_shape = [24, 24, 1]
dim_z = 100
dim_y = 5  # Cambiado a 5 según el error que mostraste

# Instanciar el modelo
gan = GAN(batch_size=batch_size, image_shape=image_shape, dim_z=dim_z, dim_y=dim_y)

# Definir las entradas
Z = tf.keras.Input(shape=(dim_z,), name='Z')
Y = tf.keras.Input(shape=(dim_y,), name='Y')
image_real = tf.keras.Input(shape=image_shape, name='image_real')

# Llamar al modelo
discrim_cost, gen_cost, p_real, p_gen = gan([Z, Y, image_real])

# Crear el modelo de Keras
model = tf.keras.Model(inputs=[Z, Y, image_real], outputs=[discrim_cost, gen_cost, p_real, p_gen])

# Compilar el modelo (ajusta los optimizadores y las funciones de pérdida según tus necesidades)
model.compile(optimizer='adam', loss='binary_crossentropy')

# Imprimir un resumen del modelo para verificar su estructura
model.summary()
