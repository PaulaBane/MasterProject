import time
from utils.data_loader import DataLoaderSF
from models.model import Model
import tensorflow as tf
import os
import numpy as np
import re
import sys

def main():
    left_img = 'Data/Sample/left/'
    right_img = 'Data/Sample/right/'
    disp_img = 'Data/Sample/disparity/'

    bat_size = 8
    # 128, 192
    maxdisp = 128
    epochs = 300
    dg = DataLoaderSF(left_img, right_img, disp_img, bat_size)

    if not os.path.exists('./results/'):
        os.mkdir('./results/')
    with tf.Session() as sess:
        PSMNet = Model(sess, height=256, weight=512, batch_size=bat_size,
                       max_disp=maxdisp, lr=0.001)
        saver = tf.train.Saver()
        for epoch in range(1, epochs + 1):
            total_train_loss = 0

            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=True)):
                start_time = time.time()
                train_loss = PSMNet.train(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d training loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss
            avg_loss = total_train_loss / (160 // bat_size)
            print('epoch %d avg training loss = %.3f' % (epoch, avg_loss))
            if epoch % 30 == 0:
                saver.save(sess, './results/PSMNet-CSPN.ckpt', global_step=epoch)

            total_train_loss = 0
            for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(dg.generator(is_training=False)):
                start_time = time.time()
                pred, train_loss = PSMNet.test(imgL_crop, imgR_crop, disp_crop_L)
                print('Iter %d testing loss = %.3f , time = %.2f' % (batch_idx, train_loss, time.time() - start_time))
                total_train_loss += train_loss

            avg_loss = total_train_loss / (40 // bat_size)
            print('epoch %d avg testing loss = %.3f' % (epoch, avg_loss))
        saver.save(sess, './results/PSMNet-CSPN.ckpt')


if __name__ == '__main__':
    main()
