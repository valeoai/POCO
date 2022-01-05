import lightconvpoint.nn as lcp_nn
import lightconvpoint.networks as lcp_net


def get_conv(backend_conv):
    """Get a convolutional layer by name.

    # Arguments
        conv_name: string.
            The name of the convolutional layer.
    """
    conv = getattr(lcp_nn, backend_conv['layer'])
    return lambda in_channels, out_channels, kernel_size: conv(in_channels, out_channels, kernel_size, **backend_conv)


def get_search(search_name):
    """Get a search algorithm by name.

    # Arguments
        search_name: string.
            The name of the search algorithm.
    """
    return getattr(lcp_nn, search_name)


def get_network(model_name, in_channels, out_channels, backend_conv, backend_search, **kwargs):
    """Get a network by name.

    # Arguments
        model_name: string.
            The name of the model.
        in_channels: int.
            The number of input channels.
        out_channels: int.
            The number of output  channels.
        ConvNet_name: string.
            The name of the convolutional layer.
        Search_name: string.
            The name of the search algorithm.
    """

    return getattr(lcp_net, model_name)(
        in_channels, out_channels, get_conv(backend_conv), get_search(backend_search), **kwargs
    )
