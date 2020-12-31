def calculate_xi_times_wij(x, w):
    return x*w


def calculate_total_layer_xi_times_wij(wij_list, xi_list):
    return sum([calculate_xi_times_wij(x=xi, w=wij) for xi, wij in zip(wij_list, xi_list)])


def calculate_hidden_layer_inputs(input, weights, bias):
    for layer, neurons in enumerate(weights):
        hidden_input_sums = []
        for neuron, weights in enumerate(neurons):
            for output_neuron, weight in enumerate(weights):
                if len(hidden_input_sums) <= output_neuron:
                    hidden_input_sums.append(bias[layer][output_neuron] + calculate_xi_times_wij(x=input[layer][neuron], w= weight))
                else:
                    hidden_input_sums [output_neuron] += calculate_xi_times_wij(x=input[layer][neuron], w= weight)
        for i in range(len(hidden_input_sums)):
            hidden_input_sums[i] = max(hidden_input_sums[i],0)
        input.append(hidden_input_sums)

    return input


def get_last_layer_weight_list(param):
    a = []
    for i in param:
        for j in i:
            a.append(j)
    return a


def calculate_lrp_z(input, weights, bias, output, layers=1):
    output_layer = output
    relevance_list = [output_layer]
    relevance_neuron_index = 0
    relevance_layer_index =0

    for layer in range(layers, -1, -1):
        layer_relevance = []
        if (len(weights) -1 ) == layer:
            wij_list = get_last_layer_weight_list(weights[layer])
            sum_of_layer_xi_times_wij = calculate_total_layer_xi_times_wij(xi_list=input[layer], wij_list=wij_list)
            for i in range(len(input[layer])):
                z_ij = calculate_xi_times_wij(input[layer][i], wij_list[i])
                relevance = z_ij * relevance_list[relevance_layer_index][relevance_neuron_index] /sum_of_layer_xi_times_wij
                layer_relevance.append(relevance)
            relevance_layer_index+=1
            relevance_neuron_index=0
            relevance_list.append(layer_relevance)
        else:
            for output_neuron, wij_list in enumerate(weights[layer]):
                sum_of_layer_xi_times_wij = calculate_total_layer_xi_times_wij(xi_list=input[layer], wij_list=wij_list)
                layer_relevance = []
                for i in range(len(input[layer])):
                    z_ij = calculate_xi_times_wij(input[layer][i], wij_list[i])
                    relevance = z_ij * relevance_list[relevance_layer_index][
                        relevance_neuron_index] / sum_of_layer_xi_times_wij
                    layer_relevance.append(relevance)

                relevance_neuron_index += 1
                relevance_list.append(layer_relevance)
            relevance_layer_index += 1
            relevance_neuron_index = 0

    print(relevance_list)
    return relevance_list


if __name__ == '__main__':

    bias = [[-1.29, 0.0], [0.0]]

    l0n0_weights = [1.3,  1.3]
    l0n1_weights = [1.2, 1.2]

    l1n0_weights = [-1.6]
    l1n1_weights = [.8]


    weights = [
        [l0n0_weights, l0n1_weights],
        [l1n0_weights, l1n1_weights]
    ]

    l1n0_input = 0.0
    l1n1_input = 1.2

    input = [
        [0.0, 1.0]
    ]

    output =  [
        0.96
               ]

    input__with_hidden_layer = calculate_hidden_layer_inputs(input, weights, bias)

    a = calculate_lrp_z(input__with_hidden_layer, weights, bias, output)
#
#
#     #
#     #
#     # #calculating the first layer relevance
#     # f_x = 0.96
#     #
#     # sum_of_layer_xi_times_wij = calculate_total_layer_xi_times_wij(xi_list=[0.0 , 1.2], wij_list=(l1n0_weights + l1n1_weights))
#     #
#     # relevance_0 =
#     # relevance_1 = f_x * (calculate_xi_times_wij(x = l1n1_input, w= l1n1_weights[0])) / sum_of_layer_xi_times_wij
