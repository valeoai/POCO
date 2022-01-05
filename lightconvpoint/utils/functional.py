import torch

def batch_gather(input, dim, index):

    index_shape = list(index.shape)
    input_shape = list(input.shape)

    views = [input.shape[0]] + [
        1 if i != dim else -1 for i in range(1, len(input.shape))
    ]
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)

    output = torch.gather(input, dim, index)

    # compute final shape
    output_shape = input_shape[0:dim] + index_shape[1:] + input_shape[dim+1:]

    return output.reshape(output_shape)