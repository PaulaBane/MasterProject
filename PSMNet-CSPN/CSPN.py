import tensorflow as tf
from tensorflow.keras.layers import Conv3D
from tensorflow.keras import Model

class Affinity_Propagate(Model):

    def __init__(self,
                 guidance,
                 blur_depth,
                 sparse_depth=None,
                 prop_time,
                 prop_kernel,
                 norm_type='8sum'):
        """

        Inputs:
            prop_time: how many steps for CSPN to perform
            prop_kernel: the size of kernel (current only support 3x3)
            way to normalize affinity
                '8sum': normalize using 8 surrounding neighborhood
                '8sum_abs': normalization enforcing affinity to be positive
                            This will lead the center affinity to be 0
        """
        super(Affinity_Propagate, self).__init__()

        self.guidance = guidance
        self.blur_depth = blur_depth
        self.sparse_depth = sparse_depth

        self.prop_time = prop_time
        self.prop_kernel = prop_kernel
        assert prop_kernel == 3, 'this version only support 8 (3x3 - 1) neighborhood'

        self.norm_type = norm_type

        self.in_feature = 1
        self.out_feature = 1

        self.forward()

    def forward(self):

        with tf.device('/device:XLA_GPU:0'):
           weight = tf.ones(shape = (1, 8, 1, 1, 1))
        self.sum_conv = Conv3D(filters = 1, 
                               kernel_size=1,
                               strides=(1, 1, 1),
                               padding='valid',
                               use_bias=False,
                               weights =[weight])
        
        gate_wb, gate_sum = self.affinity_normalization(self.guidance)

        # pad input and convert to 8 channel 3D features
        raw_depth_input = self.blur_depth

        #blur_depht_pad = nn.ZeroPad2d((1,1,1,1))
        result_depth = self.blur_depth

        if self.sparse_depth is not None:
            sparse_mask = tf.sign(self.sparse_depth)

        for i in range(self.prop_time):
            # one propagation
            spn_kernel = self.prop_kernel
            result_depth = self.pad_blur_depth(result_depth)
            neigbor_weighted_sum = self.sum_conv(gate_wb * result_depth)
            neigbor_weighted_sum = tf.squeeze(neigbor_weighted_sum, [1])
            neigbor_weighted_sum = neigbor_weighted_sum[:, :, 1:-1, 1:-1]
            result_depth = neigbor_weighted_sum

            if '8sum' in self.norm_type:
                result_depth = (1.0 - gate_sum) * raw_depth_input + result_depth
            else:
                raise ValueError('unknown norm %s' % self.norm_type)

            if self.sparse_depth is not None:
                result_depth = (1 - sparse_mask) * result_depth + sparse_mask * raw_depth_input

        return result_depth

    def affinity_normalization(self, guidance):

        # normalize features
        if 'abs' in self.norm_type:
            guidance = tf.abs(guidance)
    
        gate1_wb_cmb = guidance.narrow(1, 0                   , self.out_feature)
        gate2_wb_cmb = guidance.narrow(1, 1 * self.out_feature, self.out_feature)
        gate3_wb_cmb = guidance.narrow(1, 2 * self.out_feature, self.out_feature)
        gate4_wb_cmb = guidance.narrow(1, 3 * self.out_feature, self.out_feature)
        gate5_wb_cmb = guidance.narrow(1, 4 * self.out_feature, self.out_feature)
        gate6_wb_cmb = guidance.narrow(1, 5 * self.out_feature, self.out_feature)
        gate7_wb_cmb = guidance.narrow(1, 6 * self.out_feature, self.out_feature)
        gate8_wb_cmb = guidance.narrow(1, 7 * self.out_feature, self.out_feature)

        # gate1:left_top, gate2:center_top, gate3:right_top
        # gate4:left_center,              , gate5: right_center
        # gate6:left_bottom, gate7: center_bottom, gate8: right_bottm

        # top pad
        left_top_pad = tf.constant([[0, 2,], [0, 2]])
        gate1_wb_cmb = tf.expand_dims(tf.pad(gate1_wb_cmb, left_top_pad, "CONSTANT"), 1)

        center_top_pad = tf.constant([[0, 2], [1, 1]])
        gate2_wb_cmb = tf.expand_dims(tf.pad(gate2_wb_cmb, center_top_pad, "CONSTANT"), 1)

        right_top_pad = tf.constant([[0, 2], [2, 0]])
        gate3_wb_cmb = tf.expand_dims(tf.pad(gate3_wb_cmb, right_top_pad, "CONSTANT"), 1)

        # center pad
        left_center_pad = tf.constant([[1, 1], [0, 2]])
        gate4_wb_cmb = tf.expand_dims(tf.pad(gate4_wb_cmb, left_center_pad, "CONSTANT"), 1)

        right_center_pad = tf.constant([[1, 1], [2, 0]])
        gate5_wb_cmb = tf.expand_dims(tf.pad(gate5_wb_cmb, right_center_pad, "CONSTANT"), 1)

        # bottom pad
        left_bottom_pad = tf.constant([[2, 0], [0, 2]])
        gate6_wb_cmb = tf.expand_dims(tf.pad(gate6_wb_cmb, left_bottom_pad, "CONSTANT"), 1)

        center_bottom_pad = tf.constant([[2, 0], [1, 1]])
        gate7_wb_cmb = tf.expand_dims(tf.pad(gate7_wb_cmb, center_bottom_pad, "CONSTANT"), 1)

        right_bottm_pad = tf.constant([[2, 0], [2, 0]])
        gate8_wb_cmb = tf.expand_dims(tf.pad(gate8_wb_cmb, right_bottm_pad, "CONSTANT"), 1)

        gate_wb = tf.concat((gate1_wb_cmb,gate2_wb_cmb,gate3_wb_cmb,gate4_wb_cmb,
                             gate5_wb_cmb,gate6_wb_cmb,gate7_wb_cmb,gate8_wb_cmb), 1)

        # normalize affinity using their abs sum
        gate_wb_abs = tf.abs(gate_wb)
        abs_weight = self.sum_conv(gate_wb_abs)

        gate_wb = tf.divide(gate_wb, abs_weight)
        gate_sum = self.sum_conv(gate_wb)

        gate_sum = tf.squeeze(gate_sum, [1])
        gate_sum = gate_sum[:, :, 1:-1, 1:-1]

        return gate_wb, gate_sum

    def pad_blur_depth(self, blur_depth):
        # top pad
        left_top_pad = tf.constant([[0, 2,], [0, 2]])
        blur_depth_1 = tf.expand_dims(tf.pad(blur_depth, left_top_pad, "CONSTANT"), 1)
        center_top_pad = tf.constant([[0, 2], [1, 1]])
        blur_depth_2 = tf.expand_dims(tf.pad(blur_depth, center_top_pad, "CONSTANT"), 1)
        right_top_pad = tf.constant([[0, 2], [2, 0]])
        blur_depth_3 = tf.expand_dims(tf.pad(blur_depth, right_top_pad, "CONSTANT"), 1)

        # center pad
        left_center_pad = tf.constant([[1, 1], [0, 2]])
        blur_depth_4 = tf.expand_dims(tf.pad(blur_depth, left_center_pad, "CONSTANT"), 1)
        right_center_pad = tf.constant([[1, 1], [2, 0]])
        blur_depth_5 = tf.expand_dims(tf.pad(blur_depth, right_center_pad, "CONSTANT"), 1)

        # bottom pad
        left_bottom_pad = tf.constant([[2, 0], [0, 2]])
        blur_depth_6 = tf.expand_dims(tf.pad(blur_depth, left_bottom_pad, "CONSTANT"), 1)
        center_bottom_pad = tf.constant([[2, 0], [1, 1]])
        blur_depth_7 = tf.expand_dims(tf.pad(blur_depth, center_bottom_pad, "CONSTANT"), 1)
        right_bottm_pad = tf.constant([[2, 0], [2, 0]])
        blur_depth_8 = tf.expand_dims(tf.pad(blur_depth, right_bottm_pad, "CONSTANT"), 1)

        result_depth = tf.concat((blur_depth_1, blur_depth_2, blur_depth_3, blur_depth_4,
                                  blur_depth_5, blur_depth_6, blur_depth_7, blur_depth_8), 1)
        return result_depth

    def normalize_gate(self, guidance):
        
        gate1_x1_g1 = guidance.narrow(1,0,1)
        gate1_x1_g2 = guidance.narrow(1,1,1)
        gate1_x1_g1_abs = tf.abs(gate1_x1_g1)
        gate1_x1_g2_abs = tf.abs(gate1_x1_g2)
        elesum_gate1_x1 = tf.add(gate1_x1_g1_abs, gate1_x1_g2_abs)
        gate1_x1_g1_cmb = tf.divide(gate1_x1_g1, elesum_gate1_x1)
        gate1_x1_g2_cmb = tf.divide(gate1_x1_g2, elesum_gate1_x1)
        return gate1_x1_g1_cmb, gate1_x1_g2_cmb

    def max_of_4_tensor(self, element1, element2, element3, element4):
        max_element1_2 = tf.maximum(element1, element2)
        max_element3_4 = tf.maximum(element3, element4)
        return tf.maximum(max_element1_2, max_element3_4)

    def max_of_8_tensor(self, element1, element2, element3, element4, element5, element6, element7, element8):
        max_element1_2 = self.max_of_4_tensor(element1, element2, element3, element4)
        max_element3_4 = self.max_of_4_tensor(element5, element6, element7, element8)
        return tf.maximum(max_element1_2, max_element3_4)