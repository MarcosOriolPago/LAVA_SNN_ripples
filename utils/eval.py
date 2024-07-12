import matplotlib.pyplot as plt

class modelEval():
    def __init__(self, target, model_type, title=None):
        self.GT = target
        self.model_type = model_type
        self.title = title
        
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
        f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else 0
        
        return {
            'correct_predictions': f'{correct_predictions:.2f}',
            'total_predictions': f'{total_predictions:.2f}',
            'total_gt': f'{total_gt:.2f}',
            'precision': f'{precision:.2f}',
            'recall': f'{recall:.2f}',
            'f1_score': f'{f1_score:.2f}'
        }
    
    def plot_performance(self, performance_dict, chart):
        metrics = ['Correct Predictions', 'Total Predictions', 'Total GT', 'Precision', 'Recall', 'F1 Score']
        values = [float(performance_dict['correct_predictions']),
                float(performance_dict['total_predictions']),
                float(performance_dict['total_gt']),
                float(performance_dict['precision']),
                float(performance_dict['recall']),
                float(performance_dict['f1_score'])] # F1 score

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


        if self.title:
            fig.suptitle(self.title, fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f'{self.model_type} Performance Metrics', fontsize=14, fontweight='bold')

        if isinstance(chart, str):
            if chart.endswith('.svg'):
                plt.savefig(chart, format='svg', transparent=True)
            else:
                plt.savefig(chart)
        else:
            plt.show()