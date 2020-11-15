import tensorflow as tf
import numpy as np
import json
from typing import Generator, Dict, Type, List, Tuple, Any
import bidi.algorithm as bidi
import Levenshtein

from tfaip.base.model import ModelBase, GraphBase, ModelBaseParams
from tfaip.util.typing import AnyNumpy

from calamari_ocr.ocr.backends.dataset import CalamariData
from calamari_ocr.ocr.backends.model_interface import NetworkPredictionResult
from calamari_ocr.proto.params import ModelParams, LayerType, LayerParams
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ctc_ops as ctc

keras = tf.keras
K = keras.backend
KL = keras.layers
Model = keras.Model


def calculate_padding(input, scaling_factor):
    def scale(i, f):
        return (f - i % f) % f

    shape = tf.shape(input=input)
    px = scale(tf.gather(shape, 1), scaling_factor[0])
    py = scale(tf.gather(shape, 2), scaling_factor[1])
    return px, py


def pad(input_tensors):
    input, padding = input_tensors[0], input_tensors[1]
    px, py = padding
    shape = tf.keras.backend.shape(input)
    output = tf.image.pad_to_bounding_box(input, 0, 0, tf.keras.backend.gather(shape, 1) + px,
                                          tf.keras.backend.gather(shape, 2) + py)
    return output


class CalamariGraph(GraphBase):
    @classmethod
    def params_cls(cls):
        return ModelParams

    def __init__(self, params: ModelParams, name='CalamariGraph', **kwargs):
        super(CalamariGraph, self).__init__(params, name=name, **kwargs)

        self.conv_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        self.lstm_layers: List[Tuple[LayerParams, tf.keras.layers.Layer]] = []
        cnn_idx = 0
        for layer_index, layer in enumerate([l for l in params.layers if l.type != LayerType.LSTM]):
            if layer.type == LayerType.Convolutional:
                self.conv_layers.append((layer, KL.Conv2D(
                    name="conv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.Concat:
                self.conv_layers.append((layer, KL.Concatenate(axis=-1)))
            elif layer.type == LayerType.DilatedBlock:
                depth = max(1, layer.dilated_depth)
                dilated_layers = [
                    KL.Conv2D(
                        name='conv2d_dilated{}_{}'.format(i, cnn_idx),
                        filters=layer.filters // depth,
                        kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                        padding="same",
                        activation="relu",
                        dilation_rate=2 ** (i + 1),
                    )
                    for i in range(depth)
                ]
                concat_layer = KL.Concatenate(axis=-1)
                cnn_idx += 1
                self.conv_layers.append((layer, (dilated_layers, concat_layer)))
            elif layer.type == LayerType.TransposedConv:
                self.conv_layers.append((layer, KL.Conv2DTranspose(
                    name="tconv2d_{}".format(cnn_idx),
                    filters=layer.filters,
                    kernel_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                    activation="relu",
                )))
                cnn_idx += 1
            elif layer.type == LayerType.MaxPooling:
                self.conv_layers.append((layer, KL.MaxPool2D(
                    name="pool2d_{}".format(layer_index),
                    pool_size=(layer.kernel_size.x, layer.kernel_size.y),
                    strides=(layer.stride.x, layer.stride.y),
                    padding="same",
                )))
            else:
                raise Exception("Unknown layer of type %s" % layer.type)

        for layer_index, layer in enumerate([l for l in params.layers if l.type == LayerType.LSTM]):
            lstm = KL.LSTM(
                units=layer.hidden_nodes,
                activation='tanh',
                recurrent_activation='sigmoid',
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
                return_sequences=True,
                unit_forget_bias=True,
                name=f'lstm_{layer_index}' if layer_index > 0 else 'lstm',
            )
            self.lstm_layers.append((layer, KL.Bidirectional(
                lstm,
                name='bidirectional',
                merge_mode='concat',
            )))

        self.dropout = KL.Dropout(params.dropout)
        self.logits = KL.Dense(params.classes, name='logits')
        self.softmax = KL.Softmax(name='softmax')

    def call(self, inputs, **kwargs):
        params: ModelParams = self._params
        input_data = inputs['img']
        input_sequence_length = K.flatten(inputs['img_len'])
        shape = input_sequence_length, -1

        # if concat or conv_T layers are present, we need to pad the input to ensure that possible upsampling layers work properly
        has_concat = any([l.type == LayerType.Concat or l.type == LayerType.TransposedConv for l in params.layers])
        if has_concat:
            sx, sy = 1, 1
            for layer_index, layer in enumerate(
                    [l for l in params.layers if l.type == LayerType.MaxPooling]):
                sx *= layer.stride.x
                sy *= layer.stride.y

            padding = KL.Lambda(lambda x: calculate_padding(x, (sx, sy)), name='compute_padding')(input_data)
            padded = KL.Lambda(pad, name='padded_input')([input_data, padding])
            last_layer_output = padded
        else:
            last_layer_output = input_data

        layers_by_index = []
        for (lp, layer) in self.conv_layers:
            layers_by_index.append(last_layer_output)
            if lp.type == LayerType.Convolutional:
                last_layer_output = layer(last_layer_output)
            elif lp.type == LayerType.Concat:
                last_layer_output = layer([layers_by_index[i] for i in layer.concat_indices])
            elif lp.type == LayerType.DilatedBlock:
                dilated_layers, concat_layer = layer
                dilated_layers = [dl(last_layer_output) for dl in dilated_layers]
                last_layer_output = concat_layer(dilated_layers)
            elif lp.type == LayerType.TransposedConv:
                last_layer_output = layer(last_layer_output)
            elif lp.type == LayerType.MaxPooling:
                last_layer_output = layer(last_layer_output)
                shape = (shape[0] // lp.stride.x, shape[1] // lp.stride.y)
            else:
                raise Exception("Unknown layer of type %s" % lp.type)

        lstm_seq_len, lstm_num_features = shape
        lstm_seq_len = K.cast(lstm_seq_len, 'int32')
        ds = K.shape(last_layer_output)
        ss = last_layer_output.shape
        last_layer_output = K.reshape(last_layer_output, (ds[0], ds[1], ss[2] * ss[3]))

        if len(self.lstm_layers) > 0:
            for lstm_params, lstm_layer in self.lstm_layers:
                last_layer_output = lstm_layer(last_layer_output)

        if params.dropout > 0:
            last_layer_output = self.dropout(last_layer_output)

        logits = self.logits(last_layer_output)
        softmax = self.softmax(logits)

        blank_last_logits = tf.roll(logits, shift=-1, axis=-1)
        blank_last_softmax = tf.nn.softmax(blank_last_logits)

        greedy_decoded = ctc.ctc_greedy_decoder(inputs=array_ops.transpose(blank_last_logits, perm=[1, 0, 2]),
                                                sequence_length=tf.cast(K.flatten(lstm_seq_len),
                                                                        'int32'))[0][0]

        return {
            'blank_last_logits': blank_last_logits,
            'blank_last_softmax': blank_last_softmax,
            'out_len': lstm_seq_len,
            'logits': logits,
            'softmax': softmax,
            'decoded': tf.sparse.to_dense(greedy_decoded, default_value=-1) + 1
        }


class CalamariModel(ModelBase):
    @staticmethod
    def get_params_cls() -> Type[ModelBaseParams]:
        return ModelParams

    @classmethod
    def _get_additional_layers(cls) -> List[Type[tf.keras.layers.Layer]]:
        return [CalamariGraph]

    def __init__(self, params: ModelParams):
        super(CalamariModel, self).__init__(params)
        self._params: ModelParams = params

    def _best_logging_settings(self) -> Tuple[str, str]:
        return "min", "cer_metric"

    def create_graph(self, params: ModelBaseParams) -> 'GraphBase':
        return CalamariGraph(params)

    def _loss(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def to_2d_list(x):
            return K.expand_dims(K.flatten(x), axis=-1)

        # note: blank is last index
        loss = KL.Lambda(lambda args: K.ctc_batch_cost(args[0] - 1, args[1], args[2], args[3]), name='ctc')(
            (inputs['gt'], outputs['blank_last_softmax'], to_2d_list(outputs['out_len']), to_2d_list(inputs['gt_len'])))
        return {
            'loss': loss
        }

    def _extended_metric(self, inputs: Dict[str, tf.Tensor], outputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        def create_cer(decoded, targets, targets_length):
            greedy_decoded = tf.sparse.from_dense(decoded)
            sparse_targets = tf.cast(K.ctc_label_dense_to_sparse(targets, math_ops.cast(
                K.flatten(targets_length), dtype='int32')), 'int32')
            return tf.edit_distance(tf.cast(greedy_decoded, tf.int32), sparse_targets, normalize=True)

        # Note for codec change: the codec size is derived upon creation, therefore the ctc ops must be created
        # using the true codec size (the W/B-Matrix may change its shape however during loading/codec change
        # to match the true codec size
        cer = KL.Lambda(lambda args: create_cer(*args), output_shape=(1,), name='cer')((outputs['decoded'], inputs['gt'], inputs['gt_len']))
        return {
            'CER': cer,
        }

    def _target_prediction(self,
                           targets: Dict[str, AnyNumpy],
                           outputs: Dict[str, AnyNumpy],
                           data: 'CalamariData',
                           ) -> Tuple[Any, Any]:
        return targets['gt'], outputs['decoded'][np.where(outputs['decoded'] != -1)]

    def _print_evaluate(self, inputs: Dict[str, AnyNumpy], outputs: Dict[str, AnyNumpy], targets: Dict[str, AnyNumpy],
                        data: 'CalamariData', print_fn):
        gt, pred = self._target_prediction(targets, outputs, data)
        pred_sentence = data.params().text_post_processor.apply("".join(data.params().codec.decode(pred)))
        gt_sentence = data.params().text_post_processor.apply("".join(data.params().codec.decode(gt)))
        lr = "\u202A\u202B"
        cer = Levenshtein.distance(pred_sentence, gt_sentence) / len(gt_sentence)
        print_fn("\n  CER:  {}".format(cer) +
                 "\n  PRED: '{}{}{}'".format(lr[bidi.get_base_level(pred_sentence)], pred_sentence, "\u202C") +
                 "\n  TRUE: '{}{}{}'".format(lr[bidi.get_base_level(gt_sentence)], gt_sentence, "\u202C"))


    def predict_raw_batch(self, x: np.array, len_x: np.array) -> Generator[NetworkPredictionResult, None, None]:
        out = self.model.predict_on_batch(
            [tf.convert_to_tensor(x / 255.0, dtype=tf.float32),
             tf.convert_to_tensor(len_x, dtype=tf.int32),
             tf.zeros((len(x), 1), dtype=tf.string)],
        )
        for sm, params, sl in zip(*out):
            sl = sl[0]
            sm = np.roll(sm, 1, axis=1)
            decoded = self.ctc_decoder.decode(sm[:sl])
            pred = NetworkPredictionResult(softmax=sm,
                                           output_length=sl,
                                           decoded=decoded,
                                           )
            yield pred

    def predict_dataset(self, dataset) -> Generator[NetworkPredictionResult, None, None]:
        dataset_gen = self.create_dataset_inputs(dataset, self.batch_size, self.network_proto.features, self.network_proto.backend.shuffle_buffer_size,
                                                 mode='test')
        out = sum([list(zip(self.predict_raw_batch(d[0]['input_data'], d[0]['input_sequence_length']), d[0]['input_data_params'])) for d in dataset_gen], [])
        for pred, params in out:
            enc_param = params[0].numpy()
            pred.params = json.loads(enc_param.decode("utf-8") if isinstance(enc_param, bytes) else enc_param)
            yield pred

    def output_to_input_position(self, x):
        return x * self.scale_factor
