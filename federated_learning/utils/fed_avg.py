def average_nn_parameters(parameters):
    """
    Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params


def fed_average_nn_parameters(parameters, sizes):
    """
    Averages passed parameters on size.

    :param parameters: nn model named parameters
    :type parameters: list
    """

    new_params = {}
    sum_size = 0

    # print('size'+ str(sizes))

    for client in parameters:
        for name in parameters[client].keys():
            try:
                new_params[name].data += (parameters[client][name].data * sizes[client])
            except:
                new_params[name] = (parameters[client][name].data * sizes[client])
                # print('first agg')
        sum_size += sizes[client]

    for name in new_params:
        new_params[name].data /= sum_size

    # new_params = [new_params[name].data / sum_size for name in new_params.keys()]

    return new_params, [0]


def fedmes_average_nn_parameters(parameters, sizes):
    overlap_weight = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 3, 6: 1, 7: 2, 8: 2, 9: 1}
    new_params = {}
    sum_size_plus_overlap_weight = sum(sizes[client] + overlap_weight[client] for client in parameters.keys())

    for client, parameter in parameters.items():
        for name in parameter.keys():
            if name in new_params:
                new_params[name].data += (parameters[client][name].data * sizes[client] * overlap_weight[client])
            else:
                new_params[name] = (parameters[client][name].data * sizes[client] * overlap_weight[client])

        for name in new_params:
            new_params[name].data = new_params[name].data.float() / float(sum_size_plus_overlap_weight)

    return new_params
