import tensorflow as tf
from .conv4d import conv4d


def conv_pass(
        fmaps_in,
        kernel_size,
        num_fmaps,
        num_repetitions,
        activation='relu',
        name='conv_pass'):
    '''Create a convolution pass::

        f_in --> f_1 --> ... --> f_n

    where each ``-->`` is a convolution followed by a (non-linear) activation
    function and ``n`` ``num_repetitions``. Each convolution will decrease the
    size of the feature maps by ``kernel_size-1``.

    Args:

        f_in:

            The input tensor of shape ``(batch_size, channels, [length, ]
            depth, height, width)``.

        kernel_size:

            Size of the kernel. Forwarded to tf.layers.conv3d.

        num_fmaps:

            The number of feature maps to produce with each convolution.

        num_repetitions:

            How many convolutions to apply.

        activation:

            Which activation to use after a convolution. Accepts the name of a
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

    '''

    fmaps = fmaps_in
    if activation is not None:
        activation = getattr(tf.nn, activation)

    for i in range(num_repetitions):

        in_shape = tuple(fmaps.get_shape().as_list())

        if len(in_shape) == 6:
            conv_op = conv4d
        elif len(in_shape) == 5:
            conv_op = tf.layers.conv3d
        else:
            raise RuntimeError(
                "Input tensor of shape  % s not supported" % (in_shape, ))

        fmaps = conv_op(
            inputs=fmaps,
            filters=num_fmaps,
            kernel_size=kernel_size,
            padding='valid',
            data_format='channels_first',
            activation=activation,
            name=name + '_ % i' % i)

        out_shape = tuple(fmaps.get_shape().as_list())

        # eliminate t dimension if length is 1
        if len(out_shape) == 6:
            length = out_shape[2]
            if length == 1:
                out_shape = out_shape[0:2] + out_shape[3:]
                fmaps = tf.reshape(fmaps, out_shape)

    return fmaps


def downsample(fmaps_in, factors, name='down'):

    in_shape = fmaps_in.get_shape().as_list()
    is_4d = len(in_shape) == 6

    if is_4d:

        # store time dimension in channels
        fmaps_in = tf.reshape(fmaps_in, (
            in_shape[0],
            in_shape[1]*in_shape[2],
            in_shape[3],
            in_shape[4],
            in_shape[5]))

    fmaps = tf.layers.max_pooling3d(
        fmaps_in,
        pool_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        name=name)

    if is_4d:

        out_shape = fmaps.get_shape().as_list()

        # restore time dimension
        fmaps = tf.reshape(fmaps, (
            in_shape[0],
            in_shape[1],
            in_shape[2],
            out_shape[2],
            out_shape[3],
            out_shape[4]))

    return fmaps


def upsample(fmaps_in, factors, num_fmaps, activation='relu', name='up'):

    if activation is not None:
        activation = getattr(tf.nn, activation)

    fmaps = tf.layers.conv3d_transpose(
        fmaps_in,
        filters=num_fmaps,
        kernel_size=factors,
        strides=factors,
        padding='valid',
        data_format='channels_first',
        activation=activation,
        name=name)

    return fmaps


def crop_tzyx(fmaps_in, shape):
    '''Crop spatial and time dimensions to match shape.

    Args:

        fmaps_in:

            The input tensor of shape ``(b, c, z, y, x)`` (for 3D) or ``(b, c,
            t, z, y, x)`` (for 4D).

        shape:

            A list (not a tensor) with the requested shape ``[_, _, z, y, x]``
            (for 3D) or ``[_, _, t, z, y, x]`` (for 4D).
    '''

    in_shape = fmaps_in.get_shape().as_list()

    in_is_4d = len(in_shape) == 6
    out_is_4d = len(shape) == 6

    if in_is_4d and not out_is_4d:
        # set output shape for time to 1
        shape = shape[0:2] + [1] + shape[2:]

    if in_is_4d:
        offset = [
            0,  # batch
            0,  # channel
            (in_shape[2] - shape[2])//2,  # t
            (in_shape[3] - shape[3])//2,  # z
            (in_shape[4] - shape[4])//2,  # y
            (in_shape[5] - shape[5])//2,  # x
        ]
        size = [
            in_shape[0],
            in_shape[1],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
        ]
    else:
        offset = [
            0,  # batch
            0,  # channel
            (in_shape[2] - shape[2])//2,  # z
            (in_shape[3] - shape[3])//2,  # y
            (in_shape[4] - shape[4])//2,  # x
        ]
        size = [
            in_shape[0],
            in_shape[1],
            shape[2],
            shape[3],
            shape[4],
        ]

    fmaps = tf.slice(fmaps_in, offset, size)

    if in_is_4d and not out_is_4d:
        # remove time dimension
        shape = shape[0:2] + shape[3:]
        fmaps = tf.reshape(fmaps, shape)

    return fmaps


def unet(
        fmaps_in,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        activation='relu',
        layer=0):
    '''Create a U-Net::

        f_in --> f_left --------------------------->> f_right--> f_out
                    |                                   ^
                    v                                   |
                 g_in --> g_left ------->> g_right --> g_out
                             |               ^
                             v               |
                                   ...

    where each ``-->`` is a convolution pass (see ``conv_pass``), each `-->>` a
    crop, and down and up arrows are max-pooling and transposed convolutions,
    respectively.

    The U-Net expects 3D or 4D tensors shaped like::

        ``(batch=1, channels, [length, ] depth, height, width)``.

    This U-Net performs only "valid" convolutions, i.e., sizes of the feature
    maps decrease after each convolution. It will perfrom 4D convolutions as
    long as ``length`` is greater than 1. As soon as ``length`` is 1 due to a
    valid convolution, the time dimension will be dropped and tensors with
    ``(b, c, z, y, x)`` will be use (and returned) from there on.

    Args:

        fmaps_in:

            The input tensor.

        num_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps.
            Stored in the ``channels`` dimension.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between layers.
            If layer 0 has ``k`` feature maps, layer ``l`` will have
            ``k*fmap_inc_factor**l``.

        downsample_factors:

            List of lists ``[z, y, x]`` to use to down- and up-sample the
            feature maps between layers.

        activation:

            Which activation to use after a convolution. Accepts the name of a
            tensorflow activation function (e.g., ``relu`` for ``tf.nn.relu``).

        layer:

            Used internally to build the U-Net recursively.
    '''

    prefix = "    "*layer
    print(prefix + "Creating U-Net layer  % i" % layer)
    print(prefix + "f_in: " + str(fmaps_in.shape))

    # convolve
    f_left = conv_pass(
        fmaps_in,
        kernel_size=3,
        num_fmaps=num_fmaps,
        num_repetitions=2,
        activation=activation,
        name='unet_layer_ % i_left' % layer)

    print(prefix + "f_left: " + str(f_left.shape))

    # last layer does not recurse
    bottom_layer = (layer == len(downsample_factors))
    if bottom_layer:
        print(prefix + "bottom layer")
        print(prefix + "f_out: " + str(f_left.shape))
        return f_left

    # downsample
    g_in = downsample(
        f_left,
        downsample_factors[layer],
        'unet_down_ % i_to_ % i' % (layer, layer + 1))

    print(prefix + "g_in: " + str(g_in.shape))

    # recursive U-net
    g_out = unet(
        g_in,
        num_fmaps=num_fmaps*fmap_inc_factor,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=downsample_factors,
        activation=activation,
        layer=layer+1)

    print(prefix + "g_out: " + str(g_out.shape))

    # upsample
    g_out_upsampled = upsample(
        g_out,
        downsample_factors[layer],
        num_fmaps,
        activation=activation,
        name='unet_up_ % i_to_ % i' % (layer + 1, layer))

    print(prefix + "g_out_upsampled: " + str(g_out_upsampled.shape))

    # copy-crop
    f_left_cropped = crop_tzyx(f_left, g_out_upsampled.get_shape().as_list())

    print(prefix + "f_left_cropped: " + str(f_left_cropped.shape))

    # concatenate along channel dimension
    f_right = tf.concat([f_left_cropped, g_out_upsampled], 1)

    print(prefix + "f_right: " + str(f_right.shape))

    # convolve
    f_out = conv_pass(
        f_right,
        kernel_size=3,
        num_fmaps=num_fmaps,
        num_repetitions=2,
        name='unet_layer_ % i_right' % layer)

    print(prefix + "f_out: " + str(f_out.shape))

    return f_out


if __name__ == "__main__":

    # test
    raw = tf.placeholder(tf.float32, shape=(1, 1, 84, 268, 268))

    model = unet(raw, 12, 5, [[1, 3, 3], [1, 3, 3], [1, 3, 3]])
    tf.train.export_meta_graph(filename='unet.meta')

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())
        tf.summary.FileWriter('.', graph=tf.get_default_graph())
        # writer = tf.train.SummaryWriter(
        #       logs_path, graph=tf.get_default_graph())

    print(model.shape)
