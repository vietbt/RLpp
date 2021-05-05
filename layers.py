import numpy as np
import tensorflow as tf

def ortho_init(scale=1.0):
    def _ortho_init(shape, *_, **_kwargs):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:  # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v  # pick the one with the correct shape
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def _ln(input_tensor, gain, bias, epsilon=1e-5, axes=None):
    if axes is None:
        axes = [1]
    mean, variance = tf.nn.moments(input_tensor, axes=axes, keep_dims=True)
    input_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)
    input_tensor = input_tensor * gain + bias
    return input_tensor

def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0, layer_norm=False, act_func=None):
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable("w", [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", [n_hidden], initializer=tf.constant_initializer(init_bias))
        xw = tf.matmul(input_tensor, weight)
        if layer_norm:
            gain_x = tf.get_variable("gx", [n_hidden], initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable("bx", [n_hidden], initializer=tf.constant_initializer(0.0))
            xw = _ln(xw, gain_x, bias_x)
        if act_func:
            return act_func(xw + bias)
        else:
            return xw + bias

def mlp_extractor(input_tensor, layers, act_func=tf.nn.relu, layer_norm=False):
    output = input_tensor
    for i, layer_size in enumerate(layers):
        output = tf.layers.dense(output, layer_size, name='fc' + str(i))
        output = linear(output, f'fc{i}', layer_size, act_func=act_func, layer_norm=layer_norm)
    return output

def nature_cnn(scaled_images, **kwargs):
    act_func = tf.nn.relu
    layer_1 = act_func(conv(scaled_images, 'cnn_c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
    layer_2 = act_func(conv(layer_1, 'cnn_c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
    layer_3 = act_func(conv(layer_2, 'cnn_c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return act_func(linear(layer_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))

def conv(input_tensor, scope, *, n_filters, filter_size, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, "Filter size must have 2 elements (height, width), {} were given".format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable("w", wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable("b", bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)

def conv_to_fc(input_tensor):
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor

