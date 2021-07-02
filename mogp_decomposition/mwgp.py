"""
Copyright 2021 Siemens AG

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Authors:
Yinchong Yang <yinchong.yang@siemens.com>
Florian Buettner <buettner.florian@siemens.com>

"""

import os
import numpy as np
import tensorflow as tf
import gpflow
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

float_type = gpflow.settings.float_type


class GPDTemplate:
    def __init__(
        self,
        M,
        emb_sizes,
        batch_size=512,
        te_size=None,
        obs_mean=None,
        emb_reg=1e-3,
        lr=1e-3,
        ARD=True,
        svgp=True,
        save_path="./",
    ):
        """
        :param M: integer, number of inducing points.
        :param emb_sizes: a list of embedding sizes as integers.
        :param batch_size: integer, mini batch size for training, necessary to define the placeholders.
        :param te_size: integer, batch size for testing, necessary to define the placeholders.
        :param obs_mean: integer, mean of training targets, optional.
        :param emb_reg: float, regularization term for embeddings.
        :param lr: float, learning rate.
        :param ARD: boolean, ARD parameter in gpflow.kernel.
        :param svgp: boolean, if True then apply svgp, sgpr otherwise.
        :param save_path: string, path to save the trained models.
        """
        self.M = M
        self.emb_sizes = emb_sizes
        self.batch_size = batch_size
        self.te_size = te_size
        self.obs_mean = obs_mean
        self.emb_reg = emb_reg
        self.lr = lr
        self.ARD = ARD
        self.svgp = svgp
        self.save_path = save_path
        self.param_ids = list(locals().keys())

    def build(self):
        """
        To be initiated by children classes.
        :return: None
        """
        tf.reset_default_graph()
        gpflow.reset_default_graph_and_session()

    def save(self):
        """
        Save trained models.
        :return: None
        """
        params = {}
        for v in self.param_ids:
            if v not in ["self", "kwargs", "__class__"]:
                params[v] = self.__getattribute__(v)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        save_name = os.path.join(self.save_path, "class_params.pkl")
        with open(save_name, "wb") as handle:
            pickle.dump(params, handle)

    def load_params(self):
        """
        To be initiated by children classes.
        :return: None
        """
        return None


class GPD(GPDTemplate):
    def __init__(self, I, J, K, **kwargs):
        """
        :param I: integer, number of entities in the first dimension.
        :param J: integer, number of entities in the second dimension.
        :param K: integer, number of entities in the third dimension.
        :param kwargs: a dictionary for training hyper parameters, forwarded to GPDTemplate.__init__().
        """
        super(GPD, self).__init__(**kwargs)

        self.I = I
        self.J = J
        self.K = K
        self.param_ids.extend(list(locals().keys()))

        # Placeholders:
        self.index_1 = None
        self.index_2 = None
        self.index_3 = None
        self.y = None

        self.index_1_te = None
        self.index_2_te = None
        self.index_3_te = None

        self.index_1_M = None
        self.index_2_M = None
        self.index_3_M = None

        self.kernel_ids = None
        self.emb1 = None
        self.emb2 = None
        self.emb3 = None

        self.e1 = None
        self.e2 = None
        self.e3 = None
        self.e1_te = None
        self.e2_te = None
        self.e3_te = None
        self.e1_M = None
        self.e2_M = None
        self.e3_M = None

        self.h = None
        self.h_te = None
        self.h_M = None

        self.kernel = None
        self.gp_model = None
        self.loss = None

        self.ym = None
        self.yv = None
        self.ym_te = None
        self.yv_te = None

        self.opt_step = None
        self.sess = None

    def make_kernels(self, emb_size, kernels=None, active_dims=None):
        """
        :param emb_size: integer, embedding size for a kernel of one specific dimension.
        :param kernels: a list of strings, e.g. ['RBF', 'White], the sum of which shall form the kernel for one
        specific dimension.
        :param active_dims: active dimension of the kernel.
        :return: one kernel being the sum of all kernels required by the parameter 'kernels'.
        """
        kern = None
        if "RBF" in kernels:
            kern = gpflow.kernels.RBF(
                input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
            )

        if "Linear" in kernels:
            if kern is None:
                kern = gpflow.kernels.Linear(
                    input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
                )
            else:
                kern = kern + gpflow.kernels.Linear(
                    input_dim=emb_size, ARD=self.ARD, active_dims=active_dims
                )

        if "White" in kernels:
            kern = kern + gpflow.kernels.White(emb_size)

        return kern

    def get_kernel(self, emb_id=0, active_dims=None):
        """
        :param emb_id: string, the id of the kernel
        :param active_dims: active dims for kernel
        :return: the identified kernel
        """
        key_dict = {}
        trained_params = self.gp_model.read_trainables(self.sess)
        kern = self.make_kernels(
            kernels=self.kernel_ids,
            emb_size=self.emb_sizes[emb_id],
            active_dims=active_dims,
        )

        for i, kern__ in enumerate(self.kernel_ids):
            if kern__ == "RBF":
                key_dict["Sum/kernels/" + str(i) + "/lengthscales"] = trained_params[
                    "SVGP/kern/kernels/"
                    + str(emb_id)
                    + "/kernels/"
                    + str(i)
                    + "/lengthscales"
                ]
                key_dict["Sum/kernels/" + str(i) + "/variance"] = trained_params[
                    "SVGP/kern/kernels/"
                    + str(emb_id)
                    + "/kernels/"
                    + str(i)
                    + "/variance"
                ]
            else:
                key_dict["Sum/kernels/" + str(i) + "/variance"] = trained_params[
                    "SVGP/kern/kernels/"
                    + str(emb_id)
                    + "/kernels/"
                    + str(i)
                    + "/variance"
                ]
        kern.assign(key_dict)
        return kern

    def build_svgp(self):
        """
        Build placeholders specific for a svgp model, with deterministic batch_size.
        :return: None
        """
        print("Build SVGP")

        self.index_1 = tf.placeholder(
            tf.int64, shape=[self.batch_size, 1], name="index_1"
        )
        self.index_2 = tf.placeholder(
            tf.int64, shape=[self.batch_size, 1], name="index_2"
        )
        if self.K is not None:
            self.index_3 = tf.placeholder(
                tf.int64, shape=[self.batch_size, 1], name="index_3"
            )

        self.y = tf.placeholder(tf.float64, shape=[self.batch_size, 1], name="y")

        self.index_1_M = tf.placeholder(tf.int64, shape=[self.M, 1], name="index_1_M")
        self.index_2_M = tf.placeholder(tf.int64, shape=[self.M, 1], name="index_2_M")
        if self.K is not None:
            self.index_3_M = tf.placeholder(
                tf.int64, shape=[self.M, 1], name="index_3_M"
            )

        self.index_1_te = tf.placeholder(
            tf.int64, shape=[self.te_size, 1], name="index_1_te"
        )
        self.index_2_te = tf.placeholder(
            tf.int64, shape=[self.te_size, 1], name="index_2_te"
        )
        if self.K is not None:
            self.index_3_te = tf.placeholder(
                tf.int64, shape=[self.te_size, 1], name="index_3_te"
            )

    def build_sgpr(self):
        """
        Build placeholders specific for a sgpr model, with flexible batch_size.
        :return: None
        """
        print("Build SGPR")

        self.index_1 = tf.placeholder(tf.int64, shape=[None, 1], name="index_1")
        self.index_2 = tf.placeholder(tf.int64, shape=[None, 1], name="index_2")
        if self.K is not None:
            self.index_3 = tf.placeholder(tf.int64, shape=[None, 1], name="index_3")

        self.y = tf.placeholder(tf.float64, shape=[None, 1], name="y")

        self.index_1_M = tf.placeholder(tf.int64, shape=[None, 1], name="index_1_M")
        self.index_2_M = tf.placeholder(tf.int64, shape=[None, 1], name="index_2_M")
        if self.K is not None:
            self.index_3_M = tf.placeholder(tf.int64, shape=[None, 1], name="index_3_M")

        self.index_1_te = tf.placeholder(tf.int64, shape=[None, 1], name="index_1_te")
        self.index_2_te = tf.placeholder(tf.int64, shape=[None, 1], name="index_2_te")
        if self.K is not None:
            self.index_3_te = tf.placeholder(
                tf.int64, shape=[None, 1], name="index_3_te"
            )

    def build(self, kernels=["RBF", "White"], optimiser="adam"):  # , **kwargs
        """
        Building the GP-Decomposition model, partially by calling build_svgp or build_sgpr.
        :param kernels: list of strings, defining the kernel structure for each embedding.
        :param optimiser: string, currently either 'adam' or 'adagrad'.
        :return: None.
        """
        super(GPD, self).build()

        if self.svgp:
            self.build_svgp()
        else:
            self.build_sgpr()
        self.kernel_ids = kernels
        with tf.variable_scope("embs"):
            self.emb1 = tf.keras.layers.Embedding(
                input_dim=self.I,
                output_dim=self.emb_sizes[0],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb1",
            )
            self.emb2 = tf.keras.layers.Embedding(
                input_dim=self.J,
                output_dim=self.emb_sizes[1],
                dtype=tf.float64,
                embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                name="emb2",
            )
            if self.K is not None:
                self.emb3 = tf.keras.layers.Embedding(
                    input_dim=self.K,
                    output_dim=self.emb_sizes[2],
                    dtype=tf.float64,
                    embeddings_regularizer=tf.keras.regularizers.l2(self.emb_reg),
                    name="emb3",
                )

            self.e1 = tf.keras.layers.Flatten()(self.emb1(self.index_1))
            self.e2 = tf.keras.layers.Flatten()(self.emb2(self.index_2))
            if self.K is not None:
                self.e3 = tf.keras.layers.Flatten()(self.emb3(self.index_3))
                self.h = tf.concat([self.e1, self.e2, self.e3], axis=1)
            else:
                self.h = tf.concat([self.e1, self.e2], axis=1)

            self.e1_te = tf.keras.layers.Flatten()(self.emb1(self.index_1_te))
            self.e2_te = tf.keras.layers.Flatten()(self.emb2(self.index_2_te))
            if self.K is not None:
                self.e3_te = tf.keras.layers.Flatten()(self.emb3(self.index_3_te))
                self.h_te = tf.concat([self.e1_te, self.e2_te, self.e3_te], axis=1)
            else:
                self.h_te = tf.concat([self.e1_te, self.e2_te], axis=1)

            self.e1_M = tf.keras.layers.Flatten()(self.emb1(self.index_1_M))
            self.e2_M = tf.keras.layers.Flatten()(self.emb2(self.index_2_M))
            if self.K is not None:
                self.e3_M = tf.keras.layers.Flatten()(self.emb3(self.index_3_M))
                self.h_M = tf.concat([self.e1_M, self.e2_M, self.e3_M], axis=1)
            else:
                self.h_M = tf.concat([self.e1_M, self.e2_M], axis=1)

            self.h = tf.cast(self.h, dtype=float_type)
            self.h_te = tf.cast(self.h_te, dtype=float_type)
            self.h_M = tf.cast(self.h_M, dtype=float_type)

        # Coregionalization Kernel
        kernels1 = self.make_kernels(
            emb_size=self.emb_sizes[0],
            kernels=kernels,
            active_dims=np.arange(0, self.emb_sizes[0]),
        )

        kernels2 = self.make_kernels(
            emb_size=self.emb_sizes[1],
            kernels=kernels,
            active_dims=np.arange(
                self.emb_sizes[0], self.emb_sizes[0] + self.emb_sizes[1]
            ),
        )

        if self.K is not None:
            kernels3 = self.make_kernels(
                kernels=kernels,
                emb_size=self.emb_sizes[2],
                active_dims=np.arange(
                    self.emb_sizes[0] + self.emb_sizes[1],
                    self.emb_sizes[0] + self.emb_sizes[1] + self.emb_sizes[2],
                ),
            )
            self.kernel = kernels1 * kernels2 * kernels3
        else:
            self.kernel = kernels1 * kernels2

        if self.obs_mean is not None:
            observations_mean = tf.constant([self.obs_mean], dtype=tf.float64)
            mean_fn = lambda _: observations_mean[:, None]
        else:
            mean_fn = self.obs_mean

        Z_size = self.emb_sizes[0] + self.emb_sizes[1]
        if self.K is not None:
            Z_size = Z_size + self.emb_sizes[2]

        if self.svgp:
            self.gp_model = gpflow.models.SVGP(
                X=self.h,
                Y=tf.cast(self.y, dtype=float_type),
                Z=np.zeros((self.M, Z_size)),
                likelihood=gpflow.likelihoods.Gaussian(),
                mean_function=mean_fn,
                num_latent=1,
                kern=self.kernel,
            )
        else:
            self.gp_model = gpflow.models.SGPR(
                X=self.h,
                Y=tf.cast(self.y, dtype=float_type),
                Z=np.zeros((self.M, Z_size)),
                mean_function=mean_fn,
                kern=self.kernel,
            )

        self.loss = -self.gp_model.likelihood_tensor

        m, v = self.gp_model._build_predict(self.h)
        self.ym, self.yv = self.gp_model.likelihood.predict_mean_and_var(m, v)

        m_te, v_te = self.gp_model._build_predict(self.h_te)
        self.ym_te, self.yv_te = self.gp_model.likelihood.predict_mean_and_var(
            m_te, v_te
        )

        if optimiser == "adam":
            with tf.variable_scope("adam"):
                self.opt_step = tf.train.AdamOptimizer(
                    learning_rate=self.lr, beta1=0.0
                ).minimize(self.loss)

        else:
            with tf.variable_scope("adam"):
                self.opt_step = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

        tf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adam")
        tf_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="embs")

        self.sess = tf.Session()
        self.sess.run(tf.variables_initializer(var_list=tf_vars))
        self.gp_model.initialize(session=self.sess)
        self.train_loss = []

    def set_inducing_points(self, X_tr):
        ind = np.random.choice(range(X_tr.shape[0]), self.M, replace=False)

        if self.K is not None:
            hZ = self.sess.run(
                self.h_M,
                feed_dict={
                    self.index_1_M: X_tr[ind, 0][:, None],
                    self.index_2_M: X_tr[ind, 1][:, None],
                    self.index_3_M: X_tr[ind, 2][:, None],
                },
            )
        else:
            hZ = self.sess.run(
                self.h_M,
                feed_dict={
                    self.index_1_M: X_tr[ind, 0][:, None],
                    self.index_2_M: X_tr[ind, 1][:, None],
                },
            )

        Z_0 = hZ[np.random.choice(len(hZ), self.M, replace=False)]

        self.sess.run(
            tf.assign(
                self.gp_model.feature.Z.unconstrained_tensor,
                self.gp_model.feature.Z.transform.backward(Z_0),
            )
        )

    def train(self, X_tr, Y_tr, X_val, Y_val, n_iter):
        """
        Performs batch training of the GP decomposition model.
        :param X_tr: matrix of training input. Indices of type integer and in shape (n_triples, n_entities).
        :param Y_tr: matrix of training target. Real numbers of type float and in shape (n_triples, 1).
        :param X_val: matrix of validation input. Indices of type integer and in shape (n_triples, n_entities).
        :param Y_val: matrix of validation target. Real numbers of type float and in shape (n_triples, 1).
        :param n_iter: number of training epochs.
        :param reset_inducing: boolean, True for resetting inducing points before training.
        :return: None
        """

        self.X_tr = X_tr
        self.Y_tr = Y_tr

        mb_splits = np.array_split(
            np.arange(X_tr.shape[0]), X_tr.shape[0] // self.batch_size + 1
        )

        for l in range(n_iter):
            epoch_loss_tr = 0.0
            epoch_loss_val = 0.0
            shuffle_ids = np.random.choice(
                range(X_tr.shape[0]), X_tr.shape[0], replace=False
            )
            X_tr = np.copy(X_tr[shuffle_ids])
            Y_tr = np.copy(Y_tr[shuffle_ids])

            for ll in range(len(mb_splits)):
                mb_ids = mb_splits[ll]
                if len(mb_ids) != self.batch_size:
                    mb_ids = np.random.choice(mb_ids, self.batch_size, replace=True)

                mb_i = X_tr[mb_ids, 0]
                mb_j = X_tr[mb_ids, 1]
                if self.K is not None:
                    mb_k = X_tr[mb_ids, 2]
                mb_y = Y_tr[mb_ids]
                if self.K is not None:
                    _, mb_loss = self.sess.run(
                        [self.opt_step, self.loss],
                        feed_dict={
                            self.index_1: mb_i[:, None],
                            self.index_2: mb_j[:, None],
                            self.index_3: mb_k[:, None],
                            self.y: mb_y[:, None],
                        },
                    )
                else:
                    _, mb_loss = self.sess.run(
                        [self.opt_step, self.loss],
                        feed_dict={
                            self.index_1: mb_i[:, None],
                            self.index_2: mb_j[:, None],
                            self.y: mb_y[:, None],
                        },
                    )
                # mb_loss = mb_loss / len(mb_ids)
                epoch_loss_tr = epoch_loss_tr + mb_loss
            epoch_loss_tr = epoch_loss_tr / X_tr.shape[0]  # len(mb_splits)
            self.train_loss.append(epoch_loss_tr)

            print("epoch " + str(l) + ": " + str(epoch_loss_tr))

            if X_val is not None:
                if l % 10 == 0:
                    if self.svgp:  ## svgp:
                        hat_ids = np.random.choice(
                            X_tr.shape[0], self.batch_size, False
                        )
                        if self.K is not None:
                            y_hat = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_tr[hat_ids, 0][:, None],
                                    self.index_2_te: X_tr[hat_ids, 1][:, None],
                                    self.index_3_te: X_tr[hat_ids, 2][:, None],
                                },
                            )
                        else:
                            y_hat = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_tr[hat_ids, 0][:, None],
                                    self.index_2_te: X_tr[hat_ids, 1][:, None],
                                },
                            )
                    else:
                        hat_ids = np.random.choice(
                            X_tr.shape[0], self.batch_size, False
                        )
                        if self.K is not None:
                            y_hat = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_tr[hat_ids, 0][:, None],
                                    self.index_2_te: X_tr[hat_ids, 1][:, None],
                                    self.index_3_te: X_tr[hat_ids, 2][:, None],
                                    self.index_1: X_tr[:, 0][:, None],
                                    self.index_2: X_tr[:, 1][:, None],
                                    self.index_3: X_tr[:, 2][:, None],
                                    self.y: Y_tr[:, None],
                                },
                            )
                        else:
                            y_hat = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_tr[hat_ids, 0][:, None],
                                    self.index_2_te: X_tr[hat_ids, 1][:, None],
                                    self.index_1: X_tr[:, 0][:, None],
                                    self.index_2: X_tr[:, 1][:, None],
                                    self.y: Y_tr[:, None],
                                },
                            )

                    print(
                        "train mae:"
                        + str(
                            mean_absolute_error(
                                y_true=Y_tr[hat_ids].reshape(-1),
                                y_pred=y_hat.reshape(-1),
                            )
                        )
                    )

                    if self.svgp:
                        if self.K is not None:
                            y_pred = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_val[:, 0][:, None],
                                    self.index_2_te: X_val[:, 1][:, None],
                                    self.index_3_te: X_val[:, 2][:, None],
                                },
                            )
                        else:
                            y_pred = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_val[:, 0][:, None],
                                    self.index_2_te: X_val[:, 1][:, None],
                                },
                            )
                    else:
                        if self.K is not None:
                            y_pred = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_val[:, 0][:, None],
                                    self.index_2_te: X_val[:, 1][:, None],
                                    self.index_3_te: X_val[:, 2][:, None],
                                    self.index_1: X_tr[:, 0][:, None],
                                    self.index_2: X_tr[:, 1][:, None],
                                    self.index_3: X_tr[:, 2][:, None],
                                    self.y: Y_tr[:, None],
                                },
                            )
                        else:
                            y_pred = self.sess.run(
                                self.ym_te,
                                feed_dict={
                                    self.index_1_te: X_val[:, 0][:, None],
                                    self.index_2_te: X_val[:, 1][:, None],
                                    self.index_1: X_tr[:, 0][:, None],
                                    self.index_2: X_tr[:, 1][:, None],
                                    self.y: Y_tr[:, None],
                                },
                            )
                    print(
                        "val mae:"
                        + str(
                            mean_absolute_error(
                                y_true=Y_val.reshape(-1), y_pred=y_pred.reshape(-1)
                            )
                        )
                    )
                    print(
                        "val rmse:"
                        + str(
                            mean_squared_error(
                                y_true=Y_val.reshape(-1), y_pred=y_pred.reshape(-1)
                            )**.5
                        )
                    )
                    
    def predict(self, X_te):
        """
        Performs prediction for new indices.
        :param X_te: matrix of test input. Indices of type integer and in shape (n_triples, n_entities).
        :return: predicted values.
        """
        if self.svgp:
            if self.K is not None:
                y_pred, y_pred_var = self.sess.run(
                    [self.ym_te, self.yv_te],
                    feed_dict={
                        self.index_1_te: X_te[:, 0][:, None],
                        self.index_2_te: X_te[:, 1][:, None],
                        self.index_3_te: X_te[:, 2][:, None],
                    },
                )
            else:
                y_pred, y_pred_var = self.sess.run(
                    [self.ym_te, self.yv_te],
                    feed_dict={
                        self.index_1_te: X_te[:, 0][:, None],
                        self.index_2_te: X_te[:, 1][:, None],
                    },
                )
        else:
            if self.K is not None:
                y_pred, y_pred_var = self.sess.run(
                    [self.ym_te, self.yv_te],
                    feed_dict={
                        self.index_1_te: X_te[:, 0][:, None],
                        self.index_2_te: X_te[:, 1][:, None],
                        self.index_3_te: X_te[:, 2][:, None],
                        self.index_1: self.X_tr[:, 0][:, None],
                        self.index_2: self.X_tr[:, 1][:, None],
                        self.index_3: self.X_tr[:, 2][:, None],
                        self.y: self.Y_tr[:, None],
                    },
                )
            else:
                y_pred, y_pred_var = self.sess.run(
                    [self.ym_te, self.yv_te],
                    feed_dict={
                        self.index_1_te: X_te[:, 0][:, None],
                        self.index_2_te: X_te[:, 1][:, None],
                        self.index_1: self.X_tr[:, 0][:, None],
                        self.index_2: self.X_tr[:, 1][:, None],
                        self.y: self.Y_tr[:, None],
                    },
                )
        return [y_pred, y_pred_var]

    def get_weights_params(self):
        """
        Returns the trained model weights, parameters, as well as the hyper parameters for meta learning.
        :return: List with 4 (3) elements with last element being kernel parameters and previous elements the embeddings
        """
        super(GPD, self).save()
        embs1 = self.sess.run(self.emb1.embeddings)
        embs2 = self.sess.run(self.emb2.embeddings)
        if self.K is not None:
            embs3 = self.sess.run(self.emb3.embeddings)
        trainables = self.gp_model.read_trainables(self.sess)
        if self.K is not None:
            return [embs1, embs2, embs3, trainables]
        else:
            return [embs1, embs2, trainables]

    def save(self):
        """
        Saves the trained model weights, parameters, as well as the hyper parameters in the specified save_path.
        :return: None
        """
        super(GPD, self).save()
        embs1 = self.sess.run(self.emb1.embeddings)
        embs2 = self.sess.run(self.emb2.embeddings)
        if self.K is not None:
            embs3 = self.sess.run(self.emb3.embeddings)
        trainables = self.gp_model.read_trainables(self.sess)

        save_name = os.path.join(self.save_path, "model_params.pkl")
        if self.K is not None:
            with open(save_name, "wb") as handle:
                pickle.dump([embs1, embs2, embs3, trainables], handle)
        else:
            with open(save_name, "wb") as handle:
                pickle.dump([embs1, embs2, trainables], handle)

    def update_params(self, weights_params, epsilon=0.001):
        """
        First-order update of trained weights and parameters with fraction epsilon
        :weights_params: old weights of meta-learner
        :return: None
        """
        [embs1, embs2, embs3, gp_params] = self.get_weights_params()
        [embs1_old, embs2_old, embs3_old, gp_params_old] = weights_params

        embs1_new = embs1_old + epsilon * (embs1 - embs1_old)
        embs2_new = embs2_old + epsilon * (embs2 - embs2_old)

        gp_params_new = gp_params_old.copy()
        for key in gp_params.keys():
            gp_params_new[key] = gp_params_old[key] + epsilon * (
                gp_params[key] - gp_params_old[key]
            )

        self.gp_model.assign(gp_params_new, self.sess)
        self.sess.run(self.emb1.embeddings.assign(embs1_new))
        self.sess.run(self.emb2.embeddings.assign(embs2_new))
        if self.K is not None:
            embs3_new = embs3_old + epsilon * (embs3 - embs3_old)
            self.sess.run(self.emb3.embeddings.assign(embs3_new))

    def load_params(self, load_list=None):
        """
        Loads trained weights, parameters as well as the hyper parameters from either the specified save_path or the passed list.
        :return: None
        """
        # super(GPD, self).load_params()
        if self.K is not None:
            if load_list is None:
                # load_path = self.save_path
                load_name = os.path.join(self.save_path, "model_params.pkl")
                with open(load_name, "rb") as handle:
                    [embs1, embs2, embs3, gp_params] = pickle.load(handle)
            else:
                [embs1, embs2, embs3, gp_params] = load_list
        else:
            if load_list is None:
                load_name = os.path.join(self.save_path, "model_params.pkl")
                with open(load_name, "rb") as handle:
                    [embs1, embs2, gp_params] = pickle.load(handle)
            else:
                [embs1, embs2, gp_params] = load_list

        self.gp_model.assign(gp_params, self.sess)
        self.sess.run(self.emb1.embeddings.assign(embs1))
        self.sess.run(self.emb2.embeddings.assign(embs2))
        if self.K is not None:
            self.sess.run(self.emb3.embeddings.assign(embs3))






