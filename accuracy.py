import os
import numpy as np
import pandas as pd

def compute_accuracy(totals, predict_dir=None):

    totals.loc['Total'] = ['-','-','-','-',
                           [sum(x) for x in zip(*totals['correct_visits'])],
                           [sum(x) for x in zip(*totals['total_visits'])],
                            totals['correct_darts'].sum(),
                            totals['darts_thrown'].sum(),
                            totals['darts_predicted'].sum(),
                            totals['missed_darts'].sum(),
                            totals['extra_darts'].sum()]

    accuracies = pd.DataFrame({'dataset':totals['dataset'],'subset':totals['subset'], 'conf':totals['conf'], 'iou':totals['iou'],
                                'visit': totals.apply(lambda row: sum(row['correct_visits'])/sum(row['total_visits']), axis=1),
                                'n_dart_visits': totals.apply(lambda row: [correct/total if total!=0 else np.nan for correct, total in zip(row['correct_visits'], row['total_visits'])], axis=1),
                                'precision': totals['correct_darts'] / totals['darts_predicted'],
                                'recall': totals['correct_darts'] / totals['darts_thrown'],
                                'missed': totals['missed_darts'] / totals['darts_thrown'],
                                'extra': totals['extra_darts'] / totals['darts_predicted']})
    
    if predict_dir is not None:
        os.makedirs(predict_dir, exist_ok=True)
        
        totals.to_csv(os.path.join(predict_dir, 'totals.csv'), index=False)
        accuracies.to_csv(os.path.join(predict_dir, 'accuracies.csv'), index=False)

        totals.to_markdown(os.path.join(predict_dir, 'totals.md'))
        accuracies.to_markdown(os.path.join(predict_dir, 'accuracies.md'))
    
    return totals, accuracies
