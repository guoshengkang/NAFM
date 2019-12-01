# -*- coding:utf-8 -*-
"""
Author:
    Guosheng Kang,guoshengkang@gmail.com

Reference:
    [1] S. Rendle, “Factorization Machines,” 2010 IEEE International Conference on Data Mining, 2010, pp. 995-1000.

"""

from itertools import chain
import tensorflow as tf

from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, combined_dnn_input, DEFAULT_GROUP_NAME
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func


def basicFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], 
           l2_reg_linear=0.00001, l2_reg_embedding=0.00001, init_std=0.0001, seed=1024, task='binary'):
    """Instantiates the DeepFM Network architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
                                                                        init_std, seed, support_group=True)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    fm_logit = add_func([FM()(concat_func(v, axis=1))
                         for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit = add_func([linear_logit, fm_logit])

    output = PredictionLayer(task)(final_logit)
    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
