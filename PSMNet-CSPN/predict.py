from models.model import Model
from utils.data_loader import DataLoaderSF
import tensorflow as tf
import numpy as np
import cv2


def main():
    left_img = '../input/sceneflow-driving1/DrivingDataTest/cleanpass/left/0051.png'
    right_img = '../input/sceneflow-driving1/DrivingDataTest/cleanpass/right/0051.png'

    bat_size = 1
    maxdisp = 128

    with tf.compat.v1.Session() as sess:
        PSMNet = Model(sess, height=368, weight=1224, batch_size=bat_size, max_disp=maxdisp, lr=0.0001)
        new_saver = tf.compat.v1.train.import_meta_graph('../input/retrain/results/PSMNet.ckpt-300.meta')
        new_saver.restore(sess, tf.compat.v1.train.latest_checkpoint('../input/retrain/results/'))


        img_L = cv2.cvtColor(cv2.imread(left_img), cv2.COLOR_BGR2RGB)
        img_L = cv2.resize(img_L, (1224, 368))
        img_R = cv2.cvtColor(cv2.imread(right_img), cv2.COLOR_BGR2RGB)
        img_R = cv2.resize(img_R, (1224, 368))

        img_L = DataLoaderSF.mean_std(img_L)
        img_L = np.expand_dims(img_L, axis=0)
        img_R = DataLoaderSF.mean_std(img_R)
        img_R = np.expand_dims(img_R, axis=0)
        pred = PSMNet.predict(img_L, img_R)
        pred = np.squeeze(pred)

        item = (pred * 255 / pred.max()).astype(np.uint8)
        pred_rainbow = cv2.applyColorMap(item, cv2.COLORMAP_RAINBOW)
        cv2.imwrite('prediction.png', pred_rainbow)


if __name__ == '__main__':
    tf.compat.v1.reset_default_graph()
    main()
