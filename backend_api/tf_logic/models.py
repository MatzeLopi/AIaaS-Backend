from typing import Any
import tensorflow as tf
from tensorflow import keras
from enum import Enum

TF_ENABLE_ONEDNN_OPTS = 0


class LayerJSONError(Exception):
    pass


class LayerType(Enum):

    dense = keras.layers.Dense
    conv1d = keras.layers.Conv1D
    conv2d = keras.layers.Conv2D
    conv3d = keras.layers.Conv3D
    maxpool1d = keras.layers.MaxPool1D
    maxpool2d = keras.layers.MaxPool2D
    maxpool3d = keras.layers.MaxPool3D
    golbal_average_pooling1d = keras.layers.GlobalAveragePooling1D
    global_average_pooling2d = keras.layers.GlobalAveragePooling2D
    global_average_pooling3d = keras.layers.GlobalAveragePooling3D
    flatten = keras.layers.Flatten
    dropout = keras.layers.Dropout
    batchnorm = keras.layers.BatchNormalization
    concat = keras.layers.Concatenate
    add = keras.layers.Add


class Layer:
    pass


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


def main(layers_json):
    tf_layers:dict[str,keras.layers.Layer] = {}
    tf_inputs:dict[str,keras.layers.Layer] = {}
    tf_outputs:dict[str,keras.layers.Layer] = {}

    for lid, layer in layers_json.items():
        if layer.get("layer_type") == "Input":
            tf_inputs[lid] = layer_from_json(layer)
        elif layer.get("layer_type") == "Output":
            tf_outputs[lid] = layer_from_json(layer)
        elif layer.get("layer_type") == "Hidden":
            tf_layers[lid] = layer_from_json(layer)
        else:
            raise ValueError("Layer type not found.")

    all_layers:dict[str,keras.layers.Layer] = {**tf_inputs, **tf_layers, **tf_outputs}
    print(len(all_layers))
    print(len(tf_inputs))
    print(len(tf_layers))
    print(len(tf_outputs))

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

    print(model.summary())


if __name__ == "__main__":

    fan_in_out = {
        "layer_uuid1": {
            "tf_type": "input_layer",
            "kwargs": {"shape": 784, "name": "title", "name": "layer_uuid1"},
            "layer_type": "Input",
            "input": None,
            "output": ["layer_uuid2"],
            "uuid": "layer_uuid1",
        },
        "layer_uuid2": {
            "tf_type": "dense",
            "kwargs": {"units": 128, "activation": "relu", "name": "layer_uuid2"},
            "layer_type": "Hidden",
            "input": ["layer_uuid1"],
            "output": ["layer_uuid13"],
            "uuid": "layer_uuid2",
        },
        "layer_uuid11": {
            "tf_type": "input_layer",
            "kwargs": {"shape": 784, "name": "title", "name": "layer_uuid11"},
            "layer_type": "Input",
            "input": None,
            "output": ["layer_uuid12"],
            "uuid": "layer_uuid11",
        },
        "layer_uuid12": {
            "tf_type": "dense",
            "kwargs": {"units": 128, "activation": "relu", "name": "layer_uuid12"},
            "layer_type": "Hidden",
            "input": ["layer_uuid11"],
            "output": ["layer_uuid13"],
            "uuid": "layer_uuid12",
        },
        "layer_uuid13": {
            "tf_type": "concat",
            "kwargs": {"name": "layer_uuid13"},
            "layer_type": "Hidden",
            "input": ["layer_uuid12", "layer_uuid2"],
            "output": ["layer_uuid14"],
            "uuid": "layer_uuid13",
        },
        "layer_uuid14": {
            "tf_type": "dense",
            "kwargs": {"units": 128, "activation": "relu", "name": "layer_uuid14"},
            "layer_type": "Hidden",
            "input": ["layer_uuid13"],
            "output": ["layer_uuid15", "layer_uuid16"],
            "uuid": "layer_uuid14",
        },
        "layer_uuid15": {
            "tf_type": "dense",
            "kwargs": {"units": 10, "activation": "softmax", "name": "layer_uuid15"},
            "layer_type": "Output",
            "input": ["layer_uuid14"],
            "output": None,
            "uuid": "layer_uuid15",
        },
        "layer_uuid16": {
            "tf_type": "dense",
            "kwargs": {"units": 10, "activation": "softmax", "name": "layer_uuid16"},
            "layer_type": "Output",
            "input": ["layer_uuid14"],
            "output": None,
            "uuid": "layer_uuid16",
        },
    }
    residual = {
        "layer_uuid1": {
            "tf_type": "input_layer",
            "kwargs": {"shape": [32, 32, 3], "name": "layer_uuid1"},
            "layer_type": "Input",
            "input": None,
            "output": ["layer_uuid2"],
            "uuid": "layer_uuid1",
        },
        "layer_uuid2": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 32,
                "kernel_size": 3,
                "activation": "relu",
                "name": "layer_uuid2",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid1"],
            "output": ["layer_uuid3"],
            "uuid": "layer_uuid2",
        },
        "layer_uuid3": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "name": "layer_uuid3",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid2"],
            "output": ["layer_uuid4"],
            "uuid": "layer_uuid3",
        },
        "layer_uuid4": {
            "tf_type": "maxpool2d",
            "kwargs": {"pool_size": 3, "name": "layer_uuid4"},
            "layer_type": "Hidden",
            "input": ["layer_uuid3"],
            "output": ["layer_uuid5"],
            "uuid": "layer_uuid4",
        },
        "layer_uuid5": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "layer_uuid5",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid4"],
            "output": ["layer_uuid6"],
            "uuid": "layer_uuid5",
        },
        "layer_uuid6": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "layer_uuid6",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid5"],
            "output": ["layer_uuid7"],
            "uuid": "layer_uuid6",
        },
        "layer_uuid7": {
            "tf_type": "add",
            "kwargs": {"name": "layer_uuid7"},
            "layer_type": "Hidden",
            "input": ["layer_uuid6", "layer_uuid4"],
            "output": ["layer_uuid8"],
            "uuid": "layer_uuid7",
        },
        "layer_uuid8": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "layer_uuid8",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid7"],
            "output": ["layer_uuid9"],
            "uuid": "layer_uuid8",
        },
        "layer_uuid9": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "padding": "same",
                "name": "layer_uuid9",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid8"],
            "output": ["layer_uuid10"],
            "uuid": "layer_uuid9",
        },
        "layer_uuid10": {
            "tf_type": "add",
            "kwargs": {"name": "layer_uuid10"},
            "layer_type": "Hidden",
            "input": ["layer_uuid9", "layer_uuid7"],
            "output": ["layer_uuid11"],
            "uuid": "layer_uuid10",
        },
        "layer_uuid11": {
            "tf_type": "conv2d",
            "kwargs": {
                "filters": 64,
                "kernel_size": 3,
                "activation": "relu",
                "name": "layer_uuid11",
            },
            "layer_type": "Hidden",
            "input": ["layer_uuid10"],
            "output": ["layer_uuid12"],
            "uuid": "layer_uuid11",
        },
        "layer_uuid12": {
            "tf_type": "global_average_pooling2d",
            "kwargs": {"name": "layer_uuid12"},
            "layer_type": "Hidden",
            "input": ["layer_uuid11"],
            "output": ["layer_uuid13"],
            "uuid": "layer_uuid12",
        },
        "layer_uuid13": {
            "tf_type": "dense",
            "kwargs": {"units": 256, "activation": "relu", "name": "layer_uuid13"},
            "layer_type": "Hidden",
            "input": ["layer_uuid12"],
            "output": ["layer_uuid14"],
            "uuid": "layer_uuid13",
        },
        "layer_uuid14": {
            "tf_type": "dropout",
            "kwargs": {"rate": 0.5, "name": "layer_uuid14"},
            "layer_type": "Hidden",
            "input": ["layer_uuid13"],
            "output": ["layer_uuid15"],
            "uuid": "layer_uuid14",
        },
        "layer_uuid15": {
            "tf_type": "dense",
            "kwargs": {"units": 10, "name": "layer_uuid15"},
            "layer_type": "Output",
            "input": ["layer_uuid14"],
            "output": None,
            "uuid": "layer_uuid15",
        },
    }
    main(residual)
    inputs = keras.Input(shape=(32, 32, 3), name="img")
    x = keras.layers.Conv2D(32, 3, activation="relu")(inputs)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = keras.layers.MaxPooling2D(3)(x)

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = keras.layers.add([x, block_1_output])

    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = keras.layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = keras.layers.add([x, block_2_output])

    x = keras.layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(10)(x)

    model = keras.Model(inputs, outputs, name="toy_resnet")
    print(model.summary())
