from audioop import avg
import time
from utils.data_loader import DataLoaderSF
from models.model import Model
import tensorflow as tf
import os
import numpy as np
import re
import sys

training_losses = []
def main():
    left_img = '../input/sceneflow/Driving/cleanpass/left/'
    right_img = '../input/sceneflow/Driving/cleanpass/right/'
    disp_img = '../input/sceneflow/Driving/disparity/'

    bat_size = 8
    # 128, 192
    maxdisp = 128
    epochs = 300
    dg = DataLoaderSF(left_img, right_img, disp_img, bat_size)

    with tf.compat.v1.Session() as sess:
        PSMNet = Model(sess, height=256, weight=512, batch_size=bat_size,max_disp=maxdisp, lr=0.001, cnn_3d_type='resnet_3d')
        saver = tf.compat.v1.train.Saver()
        for epoch in range(1, epochs + 1):
            total_train_loss = 0

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                start_time = time.time()
                train_loss = PSMNet.train(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
            avg_loss = total_train_loss / (160 // bat_size)
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))
            training_losses.append(avg_loss)
            if epoch % 30 == 0:
                saver.save(sess, './results/PSMNet.ckpt', global_step=epoch)

            total_train_loss = 0
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
                start_time = time.time()
                pred, train_loss = PSMNet.test(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d testing loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss

            avg_loss = total_train_loss / (40 // bat_size)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_loss))
        saver.save(sess, './results/PSMNet.ckpt')


if __name__ == '__main__':
    tf.compat.v1.reset_default.graph()
    main()
