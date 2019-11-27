# -*- coding:utf-8 -*-
"""

"""
import tensorflow as tf
from itertools import chain
from ..inputs import input_from_feature_columns, get_linear_logit, build_input_features, DEFAULT_GROUP_NAME, combined_dnn_input
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import BiInteractionPooling
from ..layers.interaction import AFMLayer, FM
from ..layers.utils import concat_func, add_func


def NAFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(128, 128), fm_group=DEFAULT_GROUP_NAME, use_attention=True, attention_factor=8,
        l2_reg_linear=1e-5, l2_reg_embedding=1e-5, l2_reg_dnn=1e-5, l2_reg_att=1e-5, afm_dropout=0, init_std=0.0001, seed=1024, bi_dropout=0,
        dnn_dropout=0, dnn_activation='relu', task='binary'):
    """Instantiates the Attentional Factorization Machine architecture.

    :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param fm_group: list, group_name of features that will be used to do feature interactions.
    :param use_attention: bool,whether use attention or not,if set to ``False``.it is the same as **standard Factorization Machine**
    :param attention_factor: positive integer,units in attention net
    :param l2_reg_linear: float. L2 regularizer strength applied to linear part
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param l2_reg_att: float. L2 regularizer strength applied to attention net
    :param afm_dropout: float in [0,1), Fraction of the attention net output units to dropout.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :return: A Keras model instance.
    """

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())

    group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding, init_std,
                                                         seed, support_group=True)

    linear_logit = get_linear_logit(features, linear_feature_columns, init_std=init_std, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)

    sparse_embedding_list=list(chain.from_iterable(group_embedding_dict.values()))
    fm_input = concat_func(sparse_embedding_list, axis=1)
    bi_out = BiInteractionPooling()(fm_input)
    if bi_dropout:
        bi_out = tf.keras.layers.Dropout(bi_dropout)(bi_out, training=None)
    dnn_input = combined_dnn_input([bi_out], dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
                     False, seed)(dnn_input)
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, activation=None)(dnn_output)

    if use_attention:
        fm_logit = add_func([AFMLayer(attention_factor, l2_reg_att, afm_dropout,
                                      seed)(list(v)) for k, v in group_embedding_dict.items() if k in fm_group])
    else:
        fm_logit = add_func([FM()(concat_func(v, axis=1))
                             for k, v in group_embedding_dict.items() if k in fm_group])

    final_logit = add_func([linear_logit, dnn_logit, fm_logit])
    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model
