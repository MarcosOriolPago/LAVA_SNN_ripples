import torch
import h5py
import lava.lib.dl.slayer as slayer
import matplotlib.pyplot as plt


class RipplesNetwork(torch.nn.Module):
    def __init__(self, y_input_size, str_layers):
        super(RipplesNetwork, self).__init__()

        neuron_params = {
                'threshold'     : 1.25,
                'current_decay' : 0.25,
                'voltage_decay' : 0.03,
                'tau_grad'      : 0.03,
                'scale_grad'    : 3,
                'requires_grad' : True,     
            }
        neuron_params_drop = {**neuron_params, 'dropout' : slayer.neuron.Dropout(p=0.05),}
        
        module_layers = []
        raw_layers = str_layers.split('_')

        # Add the first input layer
        module_layers.append(slayer.block.cuba.Dense(neuron_params_drop, y_input_size, int(raw_layers[0]), weight_norm=True, delay=True))
        # Add the middle layers
        for i, layer in enumerate(raw_layers[1:]):
            module_layers.append(
                slayer.block.cuba.Dense(neuron_params_drop, int(raw_layers[i]), int(layer), weight_norm=True, delay=True)
            )
        # Add the last output layer
        module_layers.append(slayer.block.cuba.Dense(neuron_params, int(raw_layers[-1]), 2, weight_norm=True))

        # Add the layers as torch.nn.moduleList
        self.blocks = torch.nn.ModuleList(
                module_layers
            )
    
    def forward(self, spike):
        for block in self.blocks:
            spike = block(spike)
        return spike
    
    def grad_flow(self, path):
        # helps monitor the gradient flow
        grad = [b.synapse.grad_norm for b in self.blocks if hasattr(b, 'synapse')]

        plt.figure()
        plt.semilogy(grad)
        plt.savefig(path + 'gradFlow.png')
        plt.close()

        return grad

    def export_hdf5(self, filename):
        # network export to hdf5 format
        h = h5py.File(filename, 'w')
        layer = h.create_group('layer')
        for i, b in enumerate(self.blocks):
            b.export_hdf5(layer.create_group(f'{i}'))