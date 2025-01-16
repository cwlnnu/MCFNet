import os.path
import tensorflow as tf

from datetime import datetime
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tqdm import trange

from filtering import threshold_otsu
from decorators import image_to_tensorboard, timed
from tensorflow_addons.metrics import CohenKappa
from config import get_config
import datasets
import numpy as np
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from pdb import set_trace as bp
import tensorflow_addons as tfa
from keras import layers
from keras import models
from keras import optimizers

class classification:
    def __init__(self):
        self.PATCH_LENGTH = 3
        self.train_ratio = 0.1

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(7, 7, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(1, activation='sigmoid'))

    def sampling(self, proportion, ground_truth):
        train = {}
        test = {}
        labels_loc = {}
        m = max(ground_truth)
        for i in range(2):
            indexes = [j for j, x in enumerate(ground_truth.tolist()) if x == i]
            np.random.shuffle(indexes)
            labels_loc[i] = indexes
            if proportion != 1:
                nb_val = max(int(proportion * len(indexes)), 3)
            else:
                nb_val = 0
            train[i] = indexes[:nb_val]
            test[i] = indexes[nb_val:]
        train_indexes = []
        test_indexes = []
        for i in range(2):
            train_indexes += train[i]
            test_indexes += test[i]
        np.random.shuffle(train_indexes)
        np.random.shuffle(test_indexes)
        return train_indexes, test_indexes

    def index_assignment(self, index, row, col, pad_length):
        new_assign = {}
        for counter, value in enumerate(index):
            assign_0 = value // col + pad_length
            assign_1 = value % col + pad_length
            new_assign[counter] = [assign_0, assign_1]
        return new_assign

    def select_patch(self, matrix, pos_row, pos_col, ex_len):
        selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1)]
        selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
        return selected_patch

    def select_small_cubic(self, data_size, data_indices, whole_data, patch_length, padded_data):
        small_cubic_data = np.zeros((data_size, 2 * patch_length + 1, 2 * patch_length + 1))
        data_assign = self.index_assignment(data_indices, whole_data.shape[0], whole_data.shape[1], patch_length)
        for i in range(len(data_assign)):
            small_cubic_data[i] = self.select_patch(padded_data, data_assign[i][0], data_assign[i][1], patch_length)
        return small_cubic_data

    def __call__(self, d_f, gt):
        D_f = tf.reshape(d_f, (d_f.shape[1], d_f.shape[2]))
        GT = tf.cast(tf.reshape(gt, (gt.shape[1], gt.shape[2])), tf.float32)
        padded_D_f = np.lib.pad(D_f, ((self.PATCH_LENGTH, self.PATCH_LENGTH), (self.PATCH_LENGTH, self.PATCH_LENGTH)),
                                  'constant', constant_values=0)
        temp_D_f = tf.reshape(D_f, (np.prod(D_f.shape[:2]),))
        temp_GT1 = tf.reshape(GT, (np.prod(GT.shape[:2]),))
        temp_GT = temp_GT1.numpy()
        train_indices, test_indices = self.sampling(self.train_ratio, temp_GT)
        _, total_indices = self.sampling(1, temp_GT)

        TOTAL_SIZE = len(total_indices)
        TRAIN_SIZE = len(train_indices)

        gt_all = temp_GT[total_indices]
        y_train = temp_GT[train_indices]

        all_data = self.select_small_cubic(TOTAL_SIZE, total_indices, D_f, self.PATCH_LENGTH, padded_D_f)
        train_data = self.select_small_cubic(TRAIN_SIZE, train_indices, D_f, self.PATCH_LENGTH, padded_D_f)

        train_data_tensor = tf.convert_to_tensor(train_data, dtype=tf.float32)
        train_data_tensor = train_data_tensor[..., np.newaxis]
        y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
        all_data_tensor = tf.convert_to_tensor(all_data, dtype=tf.float32)
        all_data_tensor = all_data_tensor[..., np.newaxis]
        gt_all_tensor = tf.convert_to_tensor(gt_all, dtype=tf.float32)

        self.model.compile(optimizer=optimizers.Adam(lr=0.0005),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        self.model.fit(train_data_tensor, y_train_tensor, epochs=50, batch_size=64)

        predict = self.model.predict(all_data_tensor)
        predict = np.reshape(predict, (-1))
        y_hat = []
        for i in range(len(predict)):
            if predict[i] <= 0.5:
                y_hat.append(0)
            else:
                y_hat.append(1)
        x_label = np.zeros(temp_GT1.shape)
        x_label[total_indices] = y_hat
        x = np.ravel(x_label)
        y_re = np.reshape(x, (gt.shape[0], gt.shape[1], gt.shape[2], gt.shape[3]))

        return y_re


class ChangeDetector:
    def __init__(self, **kwargs):
        learning_rate = kwargs.get("learning_rate", 1e-4)
        lr_all = ExponentialDecay(
            learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
        )
        lr_k = ExponentialDecay(
            learning_rate, decay_steps=10000, decay_rate=0.9, staircase=True
        )
        self._optimizer_d = tf.keras.optimizers.SGD(1e-5)
        self._optimizer_g = tf.keras.optimizers.RMSprop(1e-5)

        self._optimizer_k = tf.keras.optimizers.Adam(lr_k)
        self.clipnorm = kwargs.get("clipnorm", None)

        # To keep a history for a specific training_metrics,
        # add `self.metrics_history[name] = []` in subclass __init__
        self.train_metrics = {}
        self.difference_img_metrics = {"AUC": tf.keras.metrics.AUC()}
        self.change_map_metrics = {
            "ACC": tf.keras.metrics.Accuracy(),
            "cohens kappa": CohenKappa(num_classes=2),
            # 'F1': tfa.metrics.F1Score(num_classes=2, average=None)
        }
        assert not set(self.difference_img_metrics) & set(self.change_map_metrics)
        # If the metric dictionaries shares keys, the history will not work
        self.metrics_history = {
            **{key: [] for key in self.change_map_metrics.keys()},
            **{key: [] for key in self.difference_img_metrics.keys()},
        }

        self.timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.channels = {"x": kwargs.get("channel_x"), "y": kwargs.get("channel_y")}

        # Flag used in image_to_tensorboard decorator
        self._save_images = tf.Variable(False, trainable=False)

        logdir = kwargs.get("logdir", None)
        if logdir is not None:
            self.log_path = logdir
            self.tb_writer = tf.summary.create_file_writer(self.log_path)
            self._image_dir = tf.constant(os.path.join(self.log_path, "images"))
        else:
            self.tb_writer = tf.summary.create_noop_writer()

        self.evaluation_frequency = tf.constant(
            kwargs.get("evaluation_frequency", 1), dtype=tf.int64
        )
        self.epoch = tf.Variable(0, dtype=tf.int64)
        self.stopping = tf.Variable(0, dtype=tf.int32)


    # @tf.function
    def _domain_difference_img(
        self, original, transformed, bandwidth=tf.constant(3, dtype=tf.float32)
    ):
        d = tf.norm(original - transformed, ord=2, axis=-1)
        threshold = tf.math.reduce_mean(d) + bandwidth * tf.math.reduce_std(d)
        d = tf.where(d < threshold, d, threshold)

        return tf.expand_dims(d / tf.reduce_max(d), -1)

    # @tf.function
    def _difference_img(self, x, y, x_hat, y_hat, target_change_map):
        assert x.shape[0] == y.shape[0] == 1, "Can not handle batch size > 1"

        d_x = self._domain_difference_img(x, x_hat)
        d_y = self._domain_difference_img(y, y_hat)

        temp_x = tfa.image.median_filter2d(d_x)
        temp_y = tfa.image.median_filter2d(d_y)
        temp_CMx = self._change_map(temp_x)
        temp_CMy = self._change_map(temp_y)

        true = tf.reshape(target_change_map, [-1])
        x_pred = tf.reshape(temp_CMx, [-1])
        y_pred = tf.reshape(temp_CMy, [-1])
        kappa_x = CohenKappa(num_classes=2)
        kappa_y = CohenKappa(num_classes=2)
        kappa_x.update_state(true, x_pred)
        kappa_y.update_state(true, y_pred)
        temp_x_kappa = kappa_x.result().numpy()
        temp_y_kappa = kappa_y.result().numpy()
        if temp_x_kappa > temp_y_kappa:
            d = d_x
        else:
            d = d_y

        return d

    # @tf.function
    def _change_map(self, difference_img):
        tmp = tf.cast(difference_img * 255, tf.int32)
        threshold = threshold_otsu(tmp) / 255

        return difference_img >= threshold

    @image_to_tensorboard(static_name="z_Confusion_map")
    # @tf.function
    def _confusion_map(self, target_change_map, change_map):
        """
            Compute RGB confusion map for the change map. 计算变化图的RGB混淆图。
                True positive   - White  [1,1,1]
                True negative   - Black  [0,0,0]
                False positive  - Green  [0,1,0]
                False negative  - Red    [1,0,0]
        """
        conf_map = tf.concat(
            [
                target_change_map,
                change_map,
                tf.math.logical_and(target_change_map, change_map),
            ],
            axis=-1,
            name="confusion map",
        )

        return tf.cast(conf_map, tf.float32)

    def early_stopping_criterion(self):
        """
            To be implemented in subclasses.

            Called for each epoch epoch in training. If it returns True, training will
            be terminated.

            To keep a history for a metric in self.training_metrics,
            add `self.metrics_history[name] = []` in subclass __init__
        """
        print('it is false')
        return False

    @timed
    def train(
        self,
        training_dataset,
        epochs,
        batches,
        batch_size=1,
        evaluation_dataset=None,
        filter_=None,
        final_filter=None,
        **kwargs,
    ):
        self.stopping.assign(0)
        for epoch in trange(self.epoch.numpy() + 1, self.epoch.numpy() + epochs + 1):
            self.epoch.assign(epoch)
            tf.summary.experimental.set_step(self.epoch)
            for i, batch in zip(range(batches), training_dataset.batch(batch_size)):
                self._train_step(*batch)

            with tf.device("cpu:0"):
                with self.tb_writer.as_default():
                    for name, metric in self.train_metrics.items():
                        tf.summary.scalar(name, metric.result())
                        try:
                            self.metrics_history[name].append(metric.result().numpy())
                        except KeyError as e:
                            pass
                        metric.reset_states()

            if evaluation_dataset is not None:
                for eval_data in evaluation_dataset.batch(1):
                    ev_res = self.evaluate(*eval_data, filter_)

            tf.summary.flush(self.tb_writer)
            if self.early_stopping_criterion():
                break

        self._write_metric_history()
        return self.epoch

    def evaluate(self, x, y, target_change_map, filter_=None, last=False):
        difference_img = self((x, y, target_change_map))
        if filter_ is not None:
            difference_img = filter_(self, x, y, difference_img)
            # self._ROC_curve(target_change_map, difference_img)

        self._compute_metrics(
            target_change_map, difference_img, self.difference_img_metrics
        )
        if last:
            CL = classification()
            change_map = CL(difference_img, target_change_map)
        else:
            change_map = self._change_map(difference_img)
        self._compute_metrics(target_change_map, change_map, self.change_map_metrics)

        tf.print("cohens kappa:", self.metrics_history["cohens kappa"][-1])
        tf.print("acc:", self.metrics_history["ACC"][-1])
        confusion_map = self._confusion_map(target_change_map, change_map)

        return confusion_map

    def final_evaluate(self, evaluation_dataset, save_images, final_filter, **kwargs):
        self._save_images.assign(save_images)
        for eval_data in evaluation_dataset.batch(1):
            ev_res = self.evaluate(*eval_data, final_filter, last=True)
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)

    def _compute_metrics(self, y_true, y_pred, metrics):
        y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        for name, metric in metrics.items():
            metric.update_state(y_true, y_pred)
            self.metrics_history[name].append(metric.result().numpy())

            with tf.device("cpu:0"):
                with self.tb_writer.as_default():
                    tf.summary.scalar(name, metric.result())

            metric.reset_states()

    def _write_metric_history(self):
        """ Write the contents of metrics_history to file """
        for name, history in self.metrics_history.items():
            with open(self.log_path + "/" + name + ".txt", "w") as f:
                f.write(str(history))

    @image_to_tensorboard()
    def print_image(self, x):
        return x

    def print_all_input_images(self, evaluation_dataset):
        tf.summary.experimental.set_step(self.epoch + 1)
        self._save_images.assign(True)
        for x, y, z in evaluation_dataset.batch(1):
            self.print_image(x, name="x")
            self.print_image(y, name="y")
            self.print_image(tf.cast(z, dtype=tf.float32), name="Ground_Truth")
        self._save_images.assign(False)
        tf.summary.flush(self.tb_writer)

    def save_model(self):
        print("ChangeDetector.save_model() is not implemented")

    @image_to_tensorboard(static_name="z_ROC_Curve")
    def _ROC_curve(self, y_true, y_pred):
        y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        '''fpr: [0.         0.         0.         ... 0.99999729 0.9999991  1.        ]
           tpr: [0.00000000e+00 3.03331336e-05 5.30829839e-05 ... 1.00000000e+00  1.00000000e+00 1.00000000e+00]
           _: [2.         1.         0.9996957  ... 0.09005762 0.08989964 0.08451713]'''
        roc_auc = auc(fpr, tpr)  # roc_auc: 0.19467670995619749
        fig = plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic curve")
        plt.legend(loc="lower right")
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
        data = tf.convert_to_tensor(
            data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[np.newaxis, ...],
            dtype=tf.float32,
        )
        plt.close()
        return data


def test(DATASET="Texas"):
    CONFIG = get_config(DATASET)
    _, _, EVALUATE, _ = datasets.fetch(DATASET, **CONFIG)
    cd = ChangeDetector(**CONFIG)
    cd.print_all_input_images(EVALUATE)


if __name__ == "__main__":
    test("Texas")
