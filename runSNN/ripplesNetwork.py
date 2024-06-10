import logging

from lava.magma.core.run_conditions import RunSteps
from lava.proc import io
from lava.utils.system import Loihi2

from lava.lib.dl import netx

from utils import InputAdapter, OutputAdapter, CustomSimRunConfig, np_to_event

# Define loihi2 run variables
Loihi2.preferred_partition = 'oheogulch'


def clean_output(raw_output):
    # Subtract array1 from array2
    result = raw_output[1] - raw_output[0]
    
    # Set negative values to 0
    result[result < 0] = 0
    
    return result


class loihi2Sim():
    def __init__(self, path, y_size):
        self.net = netx.hdf5.Network(net_config=path)
        self.run_config = CustomSimRunConfig()
        self.y_size = y_size
    
    def set_input(self, input):

        self.input = input
        self.event = np_to_event(input)


    def set_blocks(self):
        source = io.source.RingBuffer(data=self.event.to_tensor(dim=(1, self.input.shape[0], self.input.shape[1])).squeeze())
        self.sink = io.sink.RingBuffer(shape=self.net.out.shape, buffer=self.input.shape[1])
        inp_adapter = InputAdapter(shape=self.net.inp.shape)
        out_adapter = OutputAdapter(shape=self.net.out.shape)

        source.s_out.connect(inp_adapter.inp)
        inp_adapter.out.connect(self.net.inp)
        self.net.out.connect(out_adapter.inp)
        out_adapter.out.connect(self.sink.a_in)

    def run(self, input):
        self.set_input(input)
        self.set_blocks()

        run_condition = RunSteps(num_steps=input.shape[1])
        # self.net._log_config.level = logging.INFO
        self.net.run(condition=run_condition, run_cfg=self.run_config)
        output = self.sink.data.get()
        self.net.stop()

        return output

    def __call__(self, input):
        # Convert input to tensor if it isn't already
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)

        # Ensure the input has a batch dimension
        if input.ndimension() == 2:
            input = input.unsqueeze(0)

        raw_output = self.run(input).detach().numpy().squeeze()
        output = clean_output(raw_output)
        
        return output
        
    def show(self):
        print(self.net)
        print(f'There are {len(self.net)} layers in the network:')
        for l in self.net.layers:
            print(f'{l.block:5s} : {l.name:10s}, shape : {l.shape}')


import torch

class torchSNN():
    def __init__(self, model_path):
        self.net = torch.load(model_path)
        self.net.eval()
    

    def __call__(self, input):
        # Convert input to tensor if it isn't already
        if not isinstance(input, torch.Tensor):
            input = torch.tensor(input, dtype=torch.float32)

        # Ensure the input has a batch dimension
        if input.ndimension() == 2:
            input = input.unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            raw_output = self.net(input).detach().numpy().squeeze()
            output = clean_output(raw_output)
        
        return output

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class modelEval():
    def __init__(self, target, model_type):
        self.GT = target
        self.model_type = model_type
        
    def __call__(self, result, chart=False):
        performance = self.performance(result)
        if chart:
            self.plot_performance(performance, chart)
        else:
            return performance

    def calculate_iou(self, gt, pred):
        inter_start = max(gt[0], pred[0])
        inter_end = min(gt[1], pred[1])
        if inter_start < inter_end:
            intersection = inter_end - inter_start
        else:
            intersection = 0

        union = (gt[1] - gt[0]) + (pred[1] - pred[0]) - intersection
        iou = intersection / union
        return iou
    
    def is_match_iou(self, gt, pred, threshold=0.2):
        iou = self.calculate_iou(gt, pred)
        return iou >= threshold
    
    def performance(self, result):
        # result: predicted timings (list of (start, stop) tuples)

        correct_predictions = 0
        total_predictions = len(result)
        total_gt = len(self.GT)
        
        # Iterate over each predicted event
        for pred in result:
            if any(self.is_match_iou(gt, pred) for gt in self.GT):
                correct_predictions += 1
        
        # Calculate some metrics
        precision = correct_predictions / total_predictions if total_predictions > 0 else 0
        recall = correct_predictions / total_gt if total_gt > 0 else 0
        
        return {
            'correct_predictions': f'{correct_predictions:.2f}',
            'total_predictions': f'{total_predictions:.2f}',
            'total_gt': f'{total_gt:.2f}',
            'precision': f'{precision:.2f}',
            'recall': f'{recall:.2f}'
        }
    
    def plot_performance(self, performance_dict, chart):
        metrics = ['Correct Predictions', 'Total Predictions', 'Total GT', 'Precision', 'Recall', 'F1 Score']
        f1_score = 2 * (float(performance_dict['precision']) * float(performance_dict['recall'])) / (float(performance_dict['precision']) + float(performance_dict['recall'])) if float(performance_dict['precision']) + float(performance_dict['recall']) != 0 else 0
        values = [float(performance_dict['correct_predictions']),
                float(performance_dict['total_predictions']),
                float(performance_dict['total_gt']),
                float(performance_dict['precision']),
                float(performance_dict['recall']),
                f1_score] # F1 score

        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        plt.style.use('seaborn-v0_8-paper')

        # Bar chart for Precision and Recall
        bar_colors = ['#F8B54F', '#23CA89', '#266AC1']
        bars = ax.bar(metrics[-3:], values[-3:], color=bar_colors, width=0.5)
        ax.set_ylim([0, 1])
        # ax[0].set_xlim([-0.5,1.5])
        ax.set_ylabel('Percentage')
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Adding the data labels on top of the bars
        for bar, value in zip(bars, values[-3:]):
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, round(value, 2), ha='center', va='bottom')

        fig.suptitle(f'{self.model_type} Performance Metrics', fontsize=14, fontweight='bold')

        if isinstance(chart, str):
            if chart.endswith('.svg'):
                plt.savefig(chart, format='svg', transparent=True)
            else:
                plt.savefig(chart)
        else:
            plt.show()
    
    

