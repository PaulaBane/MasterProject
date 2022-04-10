import tensorflow as tf
import tf.keras.layers
import math

class Affinity_Propagate(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, filters=1, N=24):
        super(Affinity_Propagate, self).__init__()
        self.kernel_size=kernel_size
        self.conv2a = tf.keras.layers.Conv2D(filters, kernel_size,padding='same')
        self.conv3a = tf.keras.layers.Conv3D(filters, kernel_size=(N,kernel_size,kernel_size))
        self.N=N
    '''
    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])
    '''
    def call(self, input_tensor, training=False):
        input_tensor=tf.expand_dims(input_tensor, 0, name=None)
        input_tensor=tf.expand_dims(input_tensor, 3, name=None)
        input_tensor=tf.cast(input_tensor, dtype=tf.float32, name=None)
        output = self.conv2a(input_tensor)
        output=tf.squeeze(output, axis=[0,3], name=None)
        paddings = tf.constant([[(self.kernel_size-1)//2,(self.kernel_size-1)//2] for i in range(2)])
        output=tf.pad(output, paddings, "CONSTANT")
        output=tf.expand_dims(output, 0, name=None)
        output=tf.repeat(output, self.N, axis=0, name=None)
        output=tf.expand_dims(output, 0, name=None)
        output=tf.expand_dims(output, 4, name=None)
        output=tf.cast(output, dtype=tf.float32, name=None)
        output = self.conv3a(output)
        output=tf.squeeze(output, axis=[1,4], name=None)
        
        return output