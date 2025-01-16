import os
import gc

# Set loglevel to suppress tensorflow GPU messages
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
from PIL import Image
import tensorflow as tf
import datasets
from change_detector import ChangeDetector
from image_translation import Encoder, generator
from config import get_config_kACE
from decorators import image_to_tensorboard1, image_to_tensorboard
import numpy as np
import random


class MCFNet(ChangeDetector):
    def __init__(self, translation_spec, **kwargs):
        super().__init__(**kwargs)
        self.log = './data/'
        self.writer = tf.summary.create_file_writer(self.log)

        self._enc_x = Encoder(**translation_spec["enc_X"], name="enc_X")
        self._enc_y = Encoder(**translation_spec["enc_Y"], name="enc_Y")
        self._dec_x = generator(**translation_spec["dec_X"], name="dec_X")
        self._dec_y = generator(**translation_spec["dec_Y"], name="dec_Y")

        self.loss_object = tf.keras.losses.MeanSquaredError()

        self.train_metrics["style_x"] = tf.keras.metrics.Sum(name="style_x MSE sum")
        self.train_metrics["cross_x"] = tf.keras.metrics.Sum(name="cross_x MSE sum")
        self.train_metrics["recon_x"] = tf.keras.metrics.Sum(name="recon_x MSE sum")
        self.train_metrics["style_y"] = tf.keras.metrics.Sum(name="style_y MSE sum")
        self.train_metrics["cross_y"] = tf.keras.metrics.Sum(name="cross_y MSE sum")
        self.train_metrics["recon_y"] = tf.keras.metrics.Sum(name="recon_y MSE sum")
        self.train_metrics["l2"] = tf.keras.metrics.Sum(name="l2 MSE sum")
        self.train_metrics["total"] = tf.keras.metrics.Sum(name="total MSE sum")

        self.metrics_history["total"] = []


    def save_all_weights(self):
        self._enc_x.save_weights(self.log_path + "/weights/_enc_x/")
        self._enc_y.save_weights(self.log_path + "/weights/_enc_y/")
        self._dec_x.save_weights(self.log_path + "/weights/_dec_x/")
        self._dec_y.save_weights(self.log_path + "/weights/_dec_y/")


    def load_all_weights(self, folder):
        self._enc_x.load_weights(folder + "/weights/_enc_x/")
        self._enc_y.load_weights(folder + "/weights/_enc_y/")
        self._dec_x.load_weights(folder + "/weights/_dec_x/")
        self._dec_y.load_weights(folder + "/weights/_dec_y/")


    def enc_x(self, inputs, training=False):
        return self._enc_x(inputs, training)

    @image_to_tensorboard()
    def dec_x(self, fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, training=False):
        return self._dec_x(fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, training)

    def enc_y(self, inputs, training=False):
        return self._enc_y(inputs, training)

    @image_to_tensorboard()
    def dec_y(self, fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, training=False):
        return self._dec_y(fusion_1, fusion_2, fusion_3, fusion_4, fusion_5, training)

    def early_stopping_criterion(self):
        self.save_all_weights()
        tf.print(
            "total_loss",
            self.metrics_history["total"][-1],
        )
        return False


    def gram_matrix(self, x):
        batch_size, height, width, channels = x.shape
        x_reshaped = tf.reshape(x, (batch_size, height * width, channels))

        gram = tf.matmul(x_reshaped, x_reshaped, transpose_a=True)

        gram /= (height * width)

        return gram

    def compute_gram_matrix_per_pixel(self, x):
        B, H, W, C = x.shape

        gram_matrices = []

        for h in range(H):
            for w in range(W):
                pixel_features = x[:, h, w, :]
                pixel_features = tf.expand_dims(pixel_features, 1)
                gram_matrix = tf.matmul(pixel_features, pixel_features, transpose_a=True) / tf.cast(C, tf.float32)
                gram_matrices.append(gram_matrix)

        return tf.stack(gram_matrices, axis=1)

    def compute_mse_loss_batch(self, grams):
        B, num_grams, C, _ = grams.shape
        gram_losses = []

        for i in range(B):
            gram_list = grams[i]
            gram_loss = tf.norm(gram_list[:, tf.newaxis] - gram_list[tf.newaxis, :], axis=[-2, -1],
                                ord='euclidean') ** 2 / C
            upper_triangle = tf.linalg.band_part(gram_loss, 0, -1)
            upper_triangle_without_diag = upper_triangle - tf.linalg.band_part(gram_loss, 0, 0)
            non_zero_indices = tf.where(upper_triangle_without_diag != 0)
            non_zero_values = tf.gather_nd(upper_triangle_without_diag, non_zero_indices)
            gram_losses.append(non_zero_values)

        return tf.stack(gram_losses)


    def compute_similarity_matrix_loss(self, x, sigma=1.0):
        _, h, w, c = x.shape
        x_1 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 2)
        x_2 = tf.expand_dims(tf.reshape(x, [-1, h * w, c]), 1)
        distance = tf.norm(x_1 - x_2, axis=-1)
        similarity = np.exp(-distance ** 2 / (2 * (sigma ** 2)))
        upper_triangle = np.triu(similarity, k=1)
        non_zero_indices = np.nonzero(upper_triangle)
        non_zero_elements = upper_triangle[non_zero_indices]
        mask = tf.greater(non_zero_elements, 0.5)
        mask = tf.cast(mask, tf.float32)

        if tf.reduce_sum(mask) == 0:
            final_loss = 0.0
        else:
            grams = self.compute_gram_matrix_per_pixel(x)
            gram_losses = self.compute_mse_loss_batch(grams)
            gram_losses_flatten = tf.reshape(gram_losses, [-1])
            selected_losses = tf.multiply(mask, gram_losses_flatten)
            final_loss = tf.reduce_sum(selected_losses) / tf.reduce_sum(mask)

        return final_loss

    def combination(self, content, style):
        fusion = tf.matmul(
                    tf.reshape(content, shape=(content.shape[0], content.shape[1] * content.shape[2], -1)),
                    style)
        return tf.reshape(fusion, shape=(content.shape[0], content.shape[1], content.shape[2], -1))

    # @tf.function
    def __call__(self, inputs, training=False):
        if training:
            x, y = inputs
        else:
            x, y, target_change_map = inputs
        tf.debugging.Assert(tf.rank(x) == 4, [x.shape])
        tf.debugging.Assert(tf.rank(y) == 4, [y.shape])

        if training:
            (x_temp1, x_temp2, x_temp3, x_temp4, x_temp5), (x_content1, x_content2, x_content3, x_content4, x_content5) = self._enc_x(x, training)
            (y_temp1, y_temp2, y_temp3, y_temp4, y_temp5), (y_content1, y_content2, y_content3, y_content4, y_content5) = self._enc_y(y, training)

            loss_x_3 = self.compute_similarity_matrix_loss(x_temp5, sigma=0.8)
            loss_y_3 = self.compute_similarity_matrix_loss(y_temp5, sigma=0.8)

            matrix_loss = loss_x_3 + loss_y_3

            x_style1 = self.gram_matrix(x_temp1)
            x_style2 = self.gram_matrix(x_temp2)
            x_style3 = self.gram_matrix(x_temp3)
            x_style4 = self.gram_matrix(x_temp4)
            x_style5 = self.gram_matrix(x_temp5)

            y_style1 = self.gram_matrix(y_temp1)
            y_style2 = self.gram_matrix(y_temp2)
            y_style3 = self.gram_matrix(y_temp3)
            y_style4 = self.gram_matrix(y_temp4)
            y_style5 = self.gram_matrix(y_temp5)

            # X重建
            fusion_xx1 = self.combination(x_content1, x_style1)
            fusion_xx2 = self.combination(x_content2, x_style2)
            fusion_xx3 = self.combination(x_content3, x_style3)
            fusion_xx4 = self.combination(x_content4, x_style4)
            fusion_xx5 = self.combination(x_content5, x_style5)

            # Y重建
            fusion_yy1 = self.combination(y_content1, y_style1)
            fusion_yy2 = self.combination(y_content2, y_style2)
            fusion_yy3 = self.combination(y_content3, y_style3)
            fusion_yy4 = self.combination(y_content4, y_style4)
            fusion_yy5 = self.combination(y_content5, y_style5)

            # Y转换为X
            fusion_yx1 = self.combination(y_content1, x_style1)
            fusion_yx2 = self.combination(y_content2, x_style2)
            fusion_yx3 = self.combination(y_content3, x_style3)
            fusion_yx4 = self.combination(y_content4, x_style4)
            fusion_yx5 = self.combination(y_content5, x_style5)

            # X转换为Y
            fusion_xy1 = self.combination(x_content1, y_style1)
            fusion_xy2 = self.combination(x_content2, y_style2)
            fusion_xy3 = self.combination(x_content3, y_style3)
            fusion_xy4 = self.combination(x_content4, y_style4)
            fusion_xy5 = self.combination(x_content5, y_style5)

            # 重建图像&转换图像
            x_re = self._dec_x(fusion_xx1, fusion_xx2, fusion_xx3, fusion_xx4, fusion_xx5, training)
            x_trans = self._dec_x(fusion_yx1, fusion_yx2, fusion_yx3, fusion_yx4, fusion_yx5, training)
            if x_re.shape[1] > x.shape[1]:
                x_re = x_re[:, :-1, :, :]
            if x_re.shape[2] > x.shape[2]:
                x_re = x_re[:, :, :-1, :]
            if x_trans.shape[1] > x.shape[1]:
                x_trans = x_trans[:, :-1, :, :]
            if x_trans.shape[2] > x.shape[2]:
                x_trans = x_trans[:, :, :-1, :]
            y_re = self._dec_y(fusion_yy1, fusion_yy2, fusion_yy3, fusion_yy4, fusion_yy5, training)
            y_trans = self._dec_y(fusion_xy1, fusion_xy2, fusion_xy3, fusion_xy4, fusion_xy5, training)
            if y_re.shape[1] > y.shape[1]:
                y_re = y_re[:, :-1, :, :]
            if y_re.shape[2] > y.shape[2]:
                y_re = y_re[:, :, :-1, :]
            if y_trans.shape[1] > y.shape[1]:
                y_trans = y_trans[:, :-1, :, :]
            if y_trans.shape[2] > y.shape[2]:
                y_trans = y_trans[:, :, :-1, :]

            retval = [x_re, x_trans, y_re, y_trans, matrix_loss]

        else:
            (x_temp1, x_temp2, x_temp3, x_temp4, x_temp5), (x_content1, x_content2, x_content3, x_content4, x_content5) = self.enc_x(x)
            (y_temp1, y_temp2, y_temp3, y_temp4, y_temp5), (y_content1, y_content2, y_content3, y_content4, y_content5) = self.enc_y(y)
            x_style1 = self.gram_matrix(x_temp1)
            x_style2 = self.gram_matrix(x_temp2)
            x_style3 = self.gram_matrix(x_temp3)
            x_style4 = self.gram_matrix(x_temp4)
            x_style5 = self.gram_matrix(x_temp5)

            y_style1 = self.gram_matrix(y_temp1)
            y_style2 = self.gram_matrix(y_temp2)
            y_style3 = self.gram_matrix(y_temp3)
            y_style4 = self.gram_matrix(y_temp4)
            y_style5 = self.gram_matrix(y_temp5)

            # X重建
            fusion_xx1 = self.combination(x_content1, x_style1)
            fusion_xx2 = self.combination(x_content2, x_style2)
            fusion_xx3 = self.combination(x_content3, x_style3)
            fusion_xx4 = self.combination(x_content4, x_style4)
            fusion_xx5 = self.combination(x_content5, x_style5)

            # Y重建
            fusion_yy1 = self.combination(y_content1, y_style1)
            fusion_yy2 = self.combination(y_content2, y_style2)
            fusion_yy3 = self.combination(y_content3, y_style3)
            fusion_yy4 = self.combination(y_content4, y_style4)
            fusion_yy5 = self.combination(y_content5, y_style5)

            # Y转换为X
            fusion_yx1 = self.combination(y_content1, x_style1)
            fusion_yx2 = self.combination(y_content2, x_style2)
            fusion_yx3 = self.combination(y_content3, x_style3)
            fusion_yx4 = self.combination(y_content4, x_style4)
            fusion_yx5 = self.combination(y_content5, x_style5)

            # X转换为Y
            fusion_xy1 = self.combination(x_content1, y_style1)
            fusion_xy2 = self.combination(x_content2, y_style2)
            fusion_xy3 = self.combination(x_content3, y_style3)
            fusion_xy4 = self.combination(x_content4, y_style4)
            fusion_xy5 = self.combination(x_content5, y_style5)

            # 重建图像&转换图像
            x_re = self.dec_x(fusion_xx1, fusion_xx2, fusion_xx3, fusion_xx4, fusion_xx5, name="x_re")
            x_trans = self.dec_x(fusion_yx1, fusion_yx2, fusion_yx3, fusion_yx4, fusion_yx5, name="x_trans")
            if x_re.shape[1] > x.shape[1]:
                x_re = x_re[:, :-1, :, :]
            if x_re.shape[2] > x.shape[2]:
                x_re = x_re[:, :, :-1, :]
            if x_trans.shape[1] > x.shape[1]:
                x_trans = x_trans[:, :-1, :, :]
            if x_trans.shape[2] > x.shape[2]:
                x_trans = x_trans[:, :, :-1, :]
            y_re = self.dec_y(fusion_yy1, fusion_yy2, fusion_yy3, fusion_yy4, fusion_yy5, name="y_re")
            y_trans = self.dec_y(fusion_xy1, fusion_xy2, fusion_xy3, fusion_xy4, fusion_xy5, name="y_trans")
            if y_re.shape[1] > y.shape[1]:
                y_re = y_re[:, :-1, :, :]
            if y_re.shape[2] > y.shape[2]:
                y_re = y_re[:, :, :-1, :]
            if y_trans.shape[1] > y.shape[1]:
                y_trans = y_trans[:, :-1, :, :]
            if y_trans.shape[2] > y.shape[2]:
                y_trans = y_trans[:, :, :-1, :]
            difference_img = self._difference_img(x_re, y_re, x_trans, y_trans, target_change_map)
            retval = difference_img

        return retval

    # tf.config.run_functions_eagerly(True)

    # @tf.function
    def _train_step(self, x, y, clw):
        with tf.GradientTape() as tape:
            x_re, x_trans, y_re, y_trans, matrix_loss = self(
                [x, y], training=True
            )
            re_x_loss = 1 * self.loss_object(x, x_re)
            re_y_loss = 1 * self.loss_object(y, y_re)
            cross_x_loss = 0.2 * self.loss_object(y, y_trans, clw)
            cross_y_loss = 0.2 * self.loss_object(x, x_trans, clw)
            style_x_loss = 0.5 * self.loss_object(self.gram_matrix(x), self.gram_matrix(x_trans))
            style_y_loss = 0.5 * self.loss_object(self.gram_matrix(y), self.gram_matrix(y_trans))
            matrix_loss = 0.2 * matrix_loss
            l2_loss = (
                    sum(self._enc_x.losses)
                    + sum(self._enc_y.losses)
                    + sum(self._dec_x.losses)
                    + sum(self._dec_y.losses)
            )

            total_loss = (
                    re_x_loss
                    + cross_x_loss
                    + style_x_loss
                    + re_y_loss
                    + cross_y_loss
                    + style_y_loss
                    + l2_loss
                    + matrix_loss
            )

            targets_k = (
                    self._enc_x.trainable_variables + self._enc_y.trainable_variables + self._dec_x.trainable_variables + self._dec_y.trainable_variables
            )
            gradients_k = tape.gradient(total_loss, targets_k)
            if self.clipnorm is not None:
                gradients_k, _ = tf.clip_by_global_norm(gradients_k, self.clipnorm)

            self._optimizer_k.apply_gradients(zip(gradients_k, targets_k))

        self.train_metrics["style_x"].update_state(style_x_loss)
        self.train_metrics["cross_x"].update_state(cross_x_loss)
        self.train_metrics["recon_x"].update_state(re_x_loss)
        self.train_metrics["style_y"].update_state(style_y_loss)
        self.train_metrics["cross_y"].update_state(cross_y_loss)
        self.train_metrics["recon_y"].update_state(re_y_loss)
        self.train_metrics["l2"].update_state(l2_loss)
        self.train_metrics["total"].update_state(total_loss)


def test(DATASET="Texas", CONFIG=None):
    if CONFIG is None:
        CONFIG = get_config_kACE(DATASET)
    print(f"Loading {DATASET} data")
    x_im, y_im, EVALUATE, (C_X, C_Y) = datasets.fetch(DATASET, **CONFIG)
    if tf.config.list_physical_devices("GPU") and not CONFIG["debug"]:
        C_CODE = 3
        print("here")
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X},
            "enc_Y": {"input_chs": C_Y},
            "dec_X": {"output_chs": C_X},
            "dec_Y": {"output_chs": C_Y},
        }
    else:
        print("why here?")
        C_CODE = 1
        TRANSLATION_SPEC = {
            "enc_X": {"input_chs": C_X, "filter_spec": [C_CODE]},
            "enc_Y": {"input_chs": C_Y, "filter_spec": [C_CODE]},
            "dec_X": {"input_chs": C_CODE, "filter_spec": [C_X]},
            "dec_Y": {"input_chs": C_CODE, "filter_spec": [C_Y]},
        }
    print("Change Detector Init")
    cd = MCFNet(TRANSLATION_SPEC, **CONFIG)
    print("Training")
    t1 = time.time()
    training_time = 0
    cross_loss_weight = tf.expand_dims(tf.zeros(x_im.shape[:-1], dtype=tf.float32), -1)
    for epochs in [50, 50, 50, 50]:
        CONFIG.update(epochs=epochs)
        tr_gen, dtypes, shapes = datasets._training_data_generator(
            x_im[0], y_im[0], cross_loss_weight[0], CONFIG["patch_size"]
        )
        TRAIN = tf.data.Dataset.from_generator(tr_gen, dtypes, shapes)
        TRAIN = TRAIN.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tr_time, _ = cd.train(TRAIN, evaluation_dataset=EVALUATE, **CONFIG)
        for x, y, _ in EVALUATE.batch(1):
            alpha = cd([x, y, _])
        cross_loss_weight = 1.0 - alpha
        training_time += tr_time

    cd.load_all_weights(cd.log_path)
    cd.final_evaluate(EVALUATE, **CONFIG)
    t2 = time.time()
    final_kappa = cd.metrics_history["cohens kappa"][-1]
    print('final_kappa:', final_kappa)
    final_acc = cd.metrics_history["ACC"][-1]
    print('final_acc:', final_acc)
    print("Running time: {:.2f}".format(t2 - t1))
    performance = (final_kappa, final_acc)
    timestamp = cd.timestamp
    epoch = cd.epoch.numpy()
    speed = (epoch, training_time, timestamp)
    del cd
    gc.collect()
    return performance, speed


if __name__ == "__main__":
    # test("Texas")
    test("California")
    # test("Shuguang")