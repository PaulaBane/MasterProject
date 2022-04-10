import tensorflow as tf

 
def conv_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
               apply_bn=True, apply_relu=True):
    """Basic conv-bn-relu function"""
    with tf.compat.v1.variable_scope(name):
        conv_params = {
            'padding': 'same',
            'kernel_initializer': tf.keras.initializers.glorot_normal(),
            'kernel_regularizer': tf.keras.regularizers.L2(reg),
            'bias_regularizer': tf.keras.regularizers.L2(reg),
            'name': 'conv',
        }
        
        if dilation_rate != -1:
            conv_params['dilation_rate'] = dilation_rate
        bottom = func(filters, kernel_size, strides, **conv_params)(bottom)
        if apply_bn:
            bottom = tf.compat.v1.layers.batch_normalization(bottom,
                                                   training=tf.compat.v1.get_default_graph().get_tensor_by_name('is_training:0'),
                                                   reuse=reuse,name='bn')
                                                             
        if apply_relu:
            bottom = tf.nn.relu(bottom, name='relu')
        return bottom

def res_block(func, bottom, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None, reg=1e-4,
              projection=False):
    """Basic structure of residual block"""
    with tf.compat.v1.variable_scope(name):
        short_cut = bottom
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, name='conv1', reuse=reuse,
                            reg=reg)
        bottom = conv_block(func, bottom, filters, kernel_size, 1, dilation_rate, name='conv2', reuse=reuse, reg=reg,
                            apply_relu=False)
        if projection:
            short_cut = tf.keras.layers.Conv2D(filters, 1, strides,
                                         kernel_initializer=tf.keras.initializers.glorot_normal(),
                                         kernel_regularizer=tf.keras.regularizers.L2(reg),
                                         bias_regularizer=tf.keras.regularizers.L2(reg),
                                         name='projection')(short_cut)
        bottom = tf.add(bottom, short_cut, 'add')
        return bottom

def SPP_branch(func, bottom, pool_size, filters, kernel_size, strides=1, dilation_rate=1, name=None, reuse=None,
               reg=1e-4, apply_bn=True, apply_relu=True):
    """Spatial Pyramid Pooling function"""
    with tf.compat.v1.variable_scope(name):
        size = tf.shape(bottom)[1:3]
        bottom = tf.compat.v1.layers.average_pooling2d(bottom, pool_size, pool_size, 'same', name='avg_pool')
        bottom = conv_block(func, bottom, filters, kernel_size, strides, dilation_rate, 'conv', reuse, reg,
                            apply_bn, apply_relu)
        bottom = tf.image.resize(bottom, size)
    return bottom


def soft_arg_min(filtered_cost_volume, name):
    """Disparity Regression"""
    with tf.compat.v1.variable_scope(name):
        print('filtered_cost_volume:', filtered_cost_volume.shape)
        probability_volume = tf.nn.softmax(tf.scalar_mul(-1, filtered_cost_volume),
                                           axis=1, name='prob_volume')
        print('probability_volume:', probability_volume.shape)
        volume_shape = tf.shape(probability_volume)
        soft_1d = tf.cast(tf.range(0, volume_shape[1], dtype=tf.int32),tf.float32)
        soft_4d = tf.tile(soft_1d, tf.stack([volume_shape[0] * volume_shape[2] * volume_shape[3]]))
        soft_4d = tf.reshape(soft_4d, [volume_shape[0], volume_shape[2], volume_shape[3], volume_shape[1]])
        soft_4d = tf.transpose(soft_4d, [0, 3, 1, 2])
        estimated_disp_image = tf.reduce_sum(soft_4d * probability_volume, axis=1)
        print(estimated_disp_image.shape)
        return estimated_disp_image
