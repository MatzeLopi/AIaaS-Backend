import re
import logging
from sys import getsizeof
from uuid import uuid4
from typing import Any
from enum import Enum
from inspect import signature

import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from rapidjson import dumps

from ..constants import MODEL_DIR, LOGLEVEL

logging.basicConfig(level=LOGLEVEL)
logger = logging.getLogger(__name__)

available_layers = None


class LayerJSONError(Exception):
    pass


class LayerType(Enum):
    activation = keras.layers.Activation
    activation_regularization = keras.layers.ActivityRegularization
    add = keras.layers.Add
    additive_attention = keras.layers.AdditiveAttention
    alpha_dropout = keras.layers.AlphaDropout
    attention = keras.layers.Attention
    average = keras.layers.Average
    average_pooling_1d = keras.layers.AveragePooling1D
    average_pooling_2d = keras.layers.AveragePooling2D
    average_pooling_3d = keras.layers.AveragePooling3D
    batch_normalization = keras.layers.BatchNormalization
    bidirectional = keras.layers.Bidirectional
    catecory_encoding = keras.layers.CategoryEncoding
    center_crop = keras.layers.CenterCrop
    concatenate = keras.layers.Concatenate
    conv_1d = keras.layers.Conv1D
    conv_1d_transpose = keras.layers.Conv1DTranspose
    conv_2d = keras.layers.Conv2D
    conv_2d_transpose = keras.layers.Conv2DTranspose
    conv_3d = keras.layers.Conv3D
    conv_3d_transpose = keras.layers.Conv3DTranspose
    conv_lstm_1d = keras.layers.ConvLSTM1D
    conv_lstm_2d = keras.layers.ConvLSTM2D
    conv_lstm_3d = keras.layers.ConvLSTM3D
    cropping_1d = keras.layers.Cropping1D
    cropping_2d = keras.layers.Cropping2D
    cropping_3d = keras.layers.Cropping3D
    dense = keras.layers.Dense
    depthwise_conv_1d = keras.layers.DepthwiseConv1D
    depthwise_conv_2d = keras.layers.DepthwiseConv2D
    discretization = keras.layers.Discretization
    dot = keras.layers.Dot
    dropout = keras.layers.Dropout
    elu = keras.layers.ELU
    einsum_dense = keras.layers.EinsumDense
    embedding = keras.layers.Embedding
    flatten = keras.layers.Flatten
    flax_layer = keras.layers.FlaxLayer
    gru = keras.layers.GRU
    gru_cell = keras.layers.GRUCell
    gaussian_dropout = keras.layers.GaussianDropout
    gaussian_noise = keras.layers.GaussianNoise
    global_average_pooling_1d = keras.layers.GlobalAveragePooling1D
    global_average_pooling_2d = keras.layers.GlobalAveragePooling2D
    global_average_pooling_3d = keras.layers.GlobalAveragePooling3D
    global_max_pooling_1d = keras.layers.GlobalMaxPooling1D
    global_max_pooling_2d = keras.layers.GlobalMaxPooling2D
    global_max_pooling_3d = keras.layers.GlobalMaxPooling3D
    group_normalization = keras.layers.GroupNormalization
    group_query_attention = keras.layers.GroupNormalization
    hashed_crossing = keras.layers.HashedCrossing
    hashing = keras.layers.Hashing
    identity = keras.layers.Identity
    input_layer = keras.layers.InputLayer
    integer_lookup = keras.layers.IntegerLookup
    jax_layer = keras.layers.JaxLayer
    lstm = keras.layers.LSTM
    lstm_cell = keras.layers.LSTMCell
    lambda_layer = keras.layers.Lambda
    layer_normalization = keras.layers.LayerNormalization
    leaky_relu = keras.layers.LeakyReLU
    masking = keras.layers.Masking
    max_pool_1d = keras.layers.MaxPool1D
    max_pool_2d = keras.layers.MaxPool2D
    max_pool_3d = keras.layers.MaxPool3D
    maximum = keras.layers.Maximum
    mel_spectrogram = keras.layers.MelSpectrogram
    minimum = keras.layers.Minimum
    multi_head_attention = keras.layers.MultiHeadAttention
    multiply = keras.layers.Multiply
    normalization = keras.layers.Normalization
    p_relu = keras.layers.PReLU
    permute = keras.layers.Permute
    rnn = keras.layers.RNN
    random_brightness = keras.layers.RandomBrightness
    random_contrast = keras.layers.RandomContrast
    random_crop = keras.layers.RandomCrop
    random_flip = keras.layers.RandomFlip
    random_height = keras.layers.RandomHeight
    random_rotation = keras.layers.RandomRotation
    random_translation = keras.layers.RandomTranslation
    random_width = keras.layers.RandomWidth
    random_zoom = keras.layers.RandomZoom
    relu = keras.layers.ReLU
    repeat_vector = keras.layers.RepeatVector
    rescaling = keras.layers.Rescaling
    reshape = keras.layers.Reshape
    resizing = keras.layers.Resizing
    separable_conv_1d = keras.layers.SeparableConv1D
    separable_conv_2d = keras.layers.SeparableConv2D
    simple_rnn = keras.layers.SimpleRNN
    simple_rnn_cell = keras.layers.SimpleRNNCell
    softmax = keras.layers.Softmax
    spatial_dropout_1d = keras.layers.SpatialDropout1D
    spatial_dropout_2d = keras.layers.SpatialDropout2D
    spatial_dropout_3d = keras.layers.SpatialDropout3D
    spectral_normalization = keras.layers.SpectralNormalization
    stacked_rnn_cells = keras.layers.StackedRNNCells
    string_lookup = keras.layers.StringLookup
    subtract = keras.layers.Subtract
    tfsm_layer = keras.layers.TFSMLayer
    text_vectorization = keras.layers.TextVectorization
    thresholded_relu = keras.layers.ThresholdedReLU
    time_distributed = keras.layers.TimeDistributed
    torch_module_wrapper = keras.layers.TorchModuleWrapper
    unit_norm = keras.layers.UnitNormalization
    up_sampling_1d = keras.layers.UpSampling1D
    up_sampling_2d = keras.layers.UpSampling2D
    up_sampling_3d = keras.layers.UpSampling3D
    wrapper = keras.layers.Wrapper
    zero_padding_1d = keras.layers.ZeroPadding1D
    zero_padding_2d = keras.layers.ZeroPadding2D
    zero_padding_3d = keras.layers.ZeroPadding3D


def parse_docstring(docstring: str | None, arg_list: list[str]) -> dict:
    """
    Parses a docstring into a dictionary with argument names as keys and their descriptions as values.

    Args:
        docstring (str): The docstring to be parsed.
        arg_list (list[str]): List of arguments of the function.

    Returns:
        dict: A dictionary with argument names as keys and their descriptions as values.
    """
    # Regular expression to match the argument name and its description
    arg_pattern = re.compile(r"\s*(\w+):\s*(.*)")

    # Initialize variables
    args_dict = {}
    current_arg = None

    # Iterate over each line in the docstring
    if docstring is None:
        return {}
    for line in docstring.split("\n"):
        # Check for a match with the argument pattern
        match = arg_pattern.match(line)
        if match:
            # If a new argument is found, add it to the dictionary
            current_arg = match.group(1)
            args_dict[current_arg] = match.group(2)
        elif current_arg:
            # If the line is part of the description, append it to the current argument's description
            args_dict[current_arg] += " " + line.strip()

    # Clean up descriptions by removing excessive whitespaces
    for arg in args_dict:
        args_dict[arg] = " ".join(args_dict[arg].split())

    # Remove and args which are not in the arg_list
    args_dict = {key: value for key, value in args_dict.items() if key in arg_list}

    return args_dict


def get_args_kwargs(sig: Any) -> tuple[list[str], dict[str, str]]:
    """Get the arguments and keyword arguments of a function signature.

    Args:
        sig (Any): Function signature.

    Returns:
        tuple[list[str], dict[str, str]]: Tuple containing the arguments and keyword arguments.
    """
    # Get the parameters of the function signature
    params = sig.parameters

    # Initialize lists for arguments and keyword arguments
    args = []
    kwargs = {}

    # Iterate over the parameters
    for param in params.values():
        # If the parameter has a default value, it is a keyword argument
        if param.default != param.empty:
            kwargs[param.name] = param.default
        else:
            # Otherwise, it is a regular argument
            args.append(param.name)

    return args, kwargs


def get_arg_type(docstring: str) -> str:
    """Get the type of the argument from the docstring.

    Args:
        docstring (str): The docstring to be parsed.

    Returns:
        str: The type of the argument.
    """
    lower_doc = docstring.lower()
    if " int" in lower_doc or lower_doc.startswith("int"):
        return str(int)
    elif " str" in lower_doc or lower_doc.startswith("str"):
        return "str"
    elif " float" in lower_doc or lower_doc.startswith("float"):
        return str(float)
    elif (
        " bool" in lower_doc
        or "true" in lower_doc
        or "false" in lower_doc
        or lower_doc.startswith("bool")
    ):
        return str(bool)
    elif " list" in lower_doc or lower_doc.startswith("list"):
        return str(list)
    elif " dict" in lower_doc or lower_doc.startswith("dict"):
        return str(dict)
    else:
        return "Unknown"


def get_layers():
    """Returns a dictionary of all the layers available with their arguments and keyword arguments.

    Returns:
        dict[str[dict[str, dict[str, str]]]]: Dictionary containing the layers and their arguments.
    """
    logger.debug("Getting available layers.")
    global available_layers
    if available_layers:
        logger.debug(f"Size of cached layers: {getsizeof(available_layers)} bytes.")
        return available_layers

    layers = {}

    for layer in LayerType:
        tf_layer = layer.value
        sig = signature(tf_layer)
        params = set(sig.parameters.keys())
        args, kwargs = get_args_kwargs(sig)
        args_dict = {}
        kwargs_dict = {}

        layer_docstring: str = tf_layer.__doc__
        try:
            docstring_dict: dict[str, str] = parse_docstring(layer_docstring, params)
        except Exception as e:
            logger.error(f"Error parsing docstring for layer {layer.name}: {e}")

        else:
            docstring_dict: dict[str, dict[str, str]] = {
                arg: {"doc": doc, "type": get_arg_type(doc)}
                for arg, doc in docstring_dict.items()
            }

            args_dict = {
                arg: docstring_dict.get(arg) for arg in args if arg != "kwargs"
            }
            kwargs_dict = {kwarg: docstring_dict.get(kwarg) for kwarg in kwargs}

        layers[layer.name] = {"args": args_dict, "kwargs": kwargs_dict}

    available_layers = layers
    logger.debug(f"Layers cached. Size: {getsizeof(available_layers)} bytes.")
    return layers


def train_model(
    model: keras.Model,
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
    epochs: int = 10,
) -> keras.Model:
    """Train TF model.

    Args:
        model (keras.Model): Model to train.
        x_train (tf.Tensor): Training data.
        y_train (tf.Tensor): Training labels.
        x_test (tf.Tensor): Test data.
        y_test (tf.Tensor): Test labels.
        epochs (int): Number of epochs.

    """

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test))

    return model


def layer_from_json(layer_json: dict) -> keras.layers.Layer:
    """Create a layer from a JSON object.

    Args:
        layer_json (dict): JSON object.

    Raises:
        ValueError: If tf_type or kwargs not found in layer_json.

    Returns:
        keras.layers.Layer: Layer.

    """
    if (tf_type := layer_json.get("tf_type")) is not None:
        if tf_type == "input_layer":
            # Keras input is special case -> Creates input tensor as well as Input layer
            # Creates symbolic tensor that will be fed data later
            # Needed for the API
            kwargs = layer_json.get("kwargs")
            layer = keras.Input(**kwargs)
            return layer
        else:
            tf_type = LayerType[tf_type].value
    else:
        raise LayerJSONError("tf_type not found in layer_json.")

    if (kwargs := layer_json.get("kwargs")) is not None:
        layer = tf_type(**kwargs)
    else:
        raise LayerJSONError("kwargs not found in layer_json.")
    return layer


def create_graph(
    layer: keras.layers.Layer,
    layer_json: dict[str, Any],
    layers_json: dict[str, Any],
    all_layers: dict[str, keras.layers.Layer],
    visited: set[str],
    output_uuids: set[str],
    *,
    connected_output: list[keras.layers.Layer] | None = None,
) -> keras.layers.Layer:
    """Create the model Graph from any starting layer by traversing the JSON object which represents the model.

    Args:
        layer (keras.layers.Layer): Previous layer.
        layer_json (dict[str, Any]): JSON object of the current layer. Holds some metadata for the layer and its connections.
        layers_json (dict[str, Any]): JSON object representing the Model with all layers.
        all_layers (dict[str, keras.layers.Layer]): Dictionary of all layers.
        visited (set[str]): Set of visited layers, identified by UUID.
        output_uuids (set[str]): Set of UUIDs of the output layers.
    """
    # Check if the layer name and UUID match.
    # TODO: Probably implement checks like this earlier in the code.
    layer_name, *_ = layer.name.split("/")
    if layer_name != layer_json.get("uuid"):
        print(layer_name, layer_json.get("uuid"))
        raise LayerJSONError("Layer name and UUID do not match.")

    if connected_output is None:
        connected_output = []

    # Get the uuid of the layer
    layer_uuid = layer_json.get("uuid")
    if layer_uuid is None:
        raise LayerJSONError("Layer UUID not found in layer_json.")

    # Check if layer is input layer
    # Input layers count as visited since they are the starting point
    if layer_type := layer_json.get("layer_type"):
        if layer_type == "Input":
            # Add input layer to visited
            visited.add(layer_uuid)

    # Check if current layer has been visited
    # If is has been visited, continue with the algorithm
    # If it has not been visited, go to the previous layer
    if layer_uuid not in visited:
        previous_layer = layer_json.get("input")
        inputs = layer_json.get("input")
        # Go over the previous layers
        # Check if they have been visited
        # If they have not been visited, recursively create the graph
        for input_ in inputs:
            if input_ not in visited:
                previous_layer = all_layers[input_]
                previous_layer_json = layers_json[input_]

                return create_graph(
                    previous_layer,
                    previous_layer_json,
                    layers_json,
                    all_layers,
                    visited,
                    output_uuids,
                    connected_output=connected_output,
                )

    next_uuids = layer_json.get("output")

    # Check if the layer has an output
    if next_uuids is not None:
        if not isinstance(next_uuids, list):
            raise LayerJSONError("Output of layer is not a list.")
        # Check if the output is a single layer
        if len(next_uuids) == 1:
            [next_uuid] = next_uuids  # Get the uuid of the next layer
            next_layer: keras.layers.Layer = all_layers[next_uuid]  # Get the next layer
            next_layer_json = layers_json[
                next_uuid
            ]  # Get the JSON object of the next layer
            inputs: list[str] = next_layer_json.get(
                "input"
            )  # Get the inputs of the next layer

            # Check if all inputs are visited
            if set(inputs).issubset(visited):

                # Check if next layer has multiple inputs
                if inputs is not None and len(inputs) > 1:
                    layers = [all_layers[input_] for input_ in inputs]

                    next_layer = next_layer(layers)  # Concat layer
                    all_layers[next_uuid] = next_layer
                else:
                    next_layer = next_layer(layer)
                    all_layers[next_uuid] = next_layer
                visited.add(next_uuid)

                return create_graph(
                    next_layer,
                    next_layer_json,
                    layers_json,
                    all_layers,
                    visited,
                    output_uuids,
                    connected_output=connected_output,
                )

            # Some inputs have not been visited. Thus the network graph is not yet instantiated.
            # Walk the inputs which are not connected back until we reach layer to which we can connect or new input layer.
            else:
                # Check each input if it is visited.
                for input_ in inputs:
                    # If not visited, recursively create the graph.
                    if input_ not in visited:
                        previous_layer = all_layers[input_]
                        previous_layer_json = layers_json[input_]

                        return create_graph(
                            previous_layer,
                            previous_layer_json,
                            layers_json,
                            all_layers,
                            visited,
                            output_uuids,
                            connected_output=connected_output,
                        )

                    # Else continue to next input.
                    else:
                        continue

        # Output is multiple layers
        else:
            for next_uuid in next_uuids:
                if next_uuid in visited:
                    continue

                next_layer: keras.layers.Layer = all_layers[
                    next_uuid
                ]  # Get the next layer
                next_layer_json = layers_json[
                    next_uuid
                ]  # Get the JSON object of the next layer
                inputs: list[str] = next_layer_json.get(
                    "input"
                )  # Get the inputs of the next layer

                # Check if all inputs are visited
                if set(inputs).issubset(visited):

                    # Check if next layer has multiple inputs
                    if inputs is not None and len(inputs) > 1:
                        layers = [all_layers[input_] for input_ in inputs]

                        next_layer = next_layer(layers)  # Concat layer
                        all_layers[next_uuid] = next_layer
                    else:
                        next_layer = next_layer(layer)
                        all_layers[next_uuid] = next_layer
                    visited.add(next_uuid)

                    return create_graph(
                        next_layer,
                        next_layer_json,
                        layers_json,
                        all_layers,
                        visited,
                        output_uuids,
                        connected_output=connected_output,
                    )

            # Some inputs have not been visited. Thus the network graph is not yet instantiated.
            # Walk the inputs which are not connected back until we reach layer to which we can connect or new input layer.
            else:
                # Check each input if it is visited.
                for input_ in inputs:
                    # If not visited, recursively create the graph.
                    if input_ not in visited:
                        previous_layer = all_layers[input_]
                        previous_layer_json = layers_json[input_]

                        return create_graph(
                            previous_layer,
                            previous_layer_json,
                            layers_json,
                            all_layers,
                            visited,
                            output_uuids,
                            connected_output=connected_output,
                        )

                    # Else continue to next input.
                    else:
                        continue

    # If output layer is reached
    else:
        if isinstance(layer, keras.layers.Layer):
            pass

        elif layer.name in [
            l.name for l in connected_output
        ]:  # Fix this since symbolic tensors are not hashable.
            pass
        else:
            connected_output.append(layer)

        connected_uuids: list[str] = [
            out_layer.name.split("/")[0] for out_layer in connected_output
        ]

        if set(connected_uuids) == output_uuids:
            return connected_output
        else:
            for output_uuid in output_uuids:
                if output_uuid in connected_uuids:
                    continue
                else:
                    layer_json = layers_json[output_uuid]
                    inputs = layer_json.get("input")

                    input_, *_ = inputs
                    previous_layer = all_layers[input_]
                    previous_layer_json = layers_json[input_]
                    return create_graph(
                        previous_layer,
                        previous_layer_json,
                        layers_json,
                        all_layers,
                        visited,
                        output_uuids,
                        connected_output=connected_output,
                    )


def create_model(layers_json: dict[str, Any]) -> str:
    """Create a model from a JSON object.

    Args:
        layers_json (dict[str, Any]): JSON object representing the model.

    Raises:
        ValueError: Raises an error if the layer type is not found.

    Returns:
        str: Path to the saved model.
    """
    tf_layers: dict[str, keras.layers.Layer] = {}
    tf_inputs: dict[str, keras.layers.Layer] = {}
    tf_outputs: dict[str, keras.layers.Layer] = {}

    for lid, layer in layers_json.items():
        if layer.get("layer_type") == "Input":
            tf_inputs[lid] = layer_from_json(layer)
        elif layer.get("layer_type") == "Output":
            tf_outputs[lid] = layer_from_json(layer)
        elif layer.get("layer_type") == "Hidden":
            tf_layers[lid] = layer_from_json(layer)
        else:
            raise ValueError("Layer type not found.")

    all_layers: dict[str, keras.layers.Layer] = {**tf_inputs, **tf_layers, **tf_outputs}

    visited = set()
    input_layer, *_ = tf_layers.values()
    input_layer_json = layers_json["layer_uuid2"]
    output_layers = create_graph(
        input_layer,
        input_layer_json,
        layers_json,
        all_layers,
        visited,
        set(tf_outputs.keys()),
    )

    model = keras.Model(inputs=tf_inputs.values(), outputs=output_layers)

    model_id = uuid4().hex
    parent_folder = Path(__file__.parent).parent

    # Check if the model folder exists

    # Save the model
    model.save(MODEL_DIR / f"{model_id}.h5")

    return MODEL_DIR / f"{model_id}.h5"
