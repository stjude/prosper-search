import os

# from importlib_metadata import metadata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import seaborn as sns
import json
import sys
from sklearn.manifold import TSNE 
from itertools import product
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .analysis_lib import *

parser = argparse.ArgumentParser() 
parser.add_argument('model_name', type=str, help='Name of the model to evaluate')
parser.add_argument('agg_method', type=str, help='Aggregation method used for the model')

parser.add_argument('--topk', type = int, default = 1)
parser.add_argument('--tie_breaker_type', type=str, default='none', help='Type of generous evaluation to perform')
parser.add_argument('--remove_normal_slides', action='store_true', help='Use normal slides for evaluation')
parser.add_argument('--tumor_tiles_only', action='store_true', help='Use only tumor tiles for evaluation')

#Old keras densenet results for comparison...
#results_in = os.path.join(data_dir, f'index1024_0.2.3__metadata_0.2_mosaic_1.0_model_densenet_1.0.pkl')
#sim_in = os.path.join(data_dir, f'similarity1024_0.2.3__metadata_0.2_mosaic_1.0_model_densenet_1.0.pkl')

#CIGA
#results_in = os.path.join(data_dir, 'index512_0.2.6__metadata_0.5_mosaic_1.3.1_model_simclr-ciga_1.0.pkl')
#sim_in = os.path.join(data_dir, 'similarity512_0.2.6__metadata_0.5_mosaic_1.3.1_model_simclr-ciga_1.0.pkl')

#metadata_path = '/cnvrg/comet-pipeline/comet-test-train-split/data/output/comet_metadata_split.csv'

def eval_new_model(sim_in, metadata_path, outpath, remove_normal_slides, TOPK, tie_breaker_type):
    print(outpath)
    os.makedirs(outpath, exist_ok = True)

    ##New indices: (0.2.6+)
    sim = load_results_new(sim_in)
    all_metadata = load_metadata(metadata_path)
    # all_metadata['Tumor.group'] = all_metadata['Tumor.group'].replace(['ACT', 'ACC'], 'ACT/ACC')

    # print(all_metadata)
    sim_filtered, slides_filtered, all_metadata = filter_data_new(sim, all_metadata, remove_normal_slides)
    # np.save(f'{data_dir}/to_Ali/sim_filtered.npy', sim_filtered)
    # np.save(f'{data_dir}/to_Ali/slides_filtered.npy', slides_filtered)
    # sim_filtered.to_pickle(f'{data_dir}/to_Ali/sim_filtered.pkl')
    # all_metadata.to_pickle(f'{data_dir}/to_Ali/all_metadata.pkl')
    # print(sim_filtered)
    # results = make_weighted_nn_predictions(all_metadata, slides_filtered, sim_filtered, baseline)
    if tie_breaker_type == 'absolute':
        results = make_nn_predictions_tie_breaker_absolute(all_metadata, slides_filtered, sim_filtered, topk = TOPK)
    elif tie_breaker_type == 'generous':
        results = make_nn_predictions_tie_breaker_generous(all_metadata, slides_filtered, sim_filtered, topk = TOPK)
    elif tie_breaker_type == 'closest':
        results = make_nn_predictions_tie_breaker_closest(all_metadata, slides_filtered, sim_filtered, topk = TOPK)
    elif tie_breaker_type == 'binary':
        results = make_nn_predictions_tie_breaker_binary(all_metadata, slides_filtered, sim_filtered, topk = TOPK)
        
    results.to_csv(f'{outpath}/pred_results.csv')
    results.to_pickle(f'{outpath}/pred_results.pkl')
    # print(f'{data_dir}/fig/{model_name}/pred_results.csv')

    from sklearn.metrics import balanced_accuracy_score

    y_true = results['query_class']
    y_pred = results['prediction']

    balanced_acc = balanced_accuracy_score(y_true, y_pred)                 # = macro recall
    balanced_acc_adj = balanced_accuracy_score(y_true, y_pred, adjusted=True)  # chance-adjusted


    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average = 'macro')
    pr = precision_score(y_true, y_pred, average = 'macro')
    re = recall_score(y_true, y_pred, average = 'macro')


    print('accuracy:', acc)
    print('f1:', f1)
    print('precision:', pr)
    print('recall:', re)
    print('balanced accuracy:', balanced_acc)

    # Save evaluation metrics to a JSON file
    metrics = {
        'accuracy': acc,
        'f1': f1,
        'precision': pr,
        'recall': re,
        'balanced_accuracy': balanced_acc,
    }

    with open(f'{outpath}/evaluation_metrics.json', 'w') as json_file:
        json.dump(metrics, json_file, indent=4)


    # diag_counts = [(k,v['Tumor.group']) for k,v in all_metadata['Tumor.group'].value_counts().reset_index().set_index('index').to_dict('index').items()]
    # diag_counts = [(k,v['Tumor.group']) for k,v in all_metadata['Tumor.group'].value_counts().reset_index()]
    # diag_counts = [(k,v['Tumor.group']) for k,v in all_metadata['Tumor.group'].value_counts().reset_index().rename(columns={'index': 'new_index'}).set_index('new_index').to_dict('index').items()]     
    # diag_counts = sorted(diag_counts, key = lambda x: x[1], reverse = True)
    # diag_sorted = [d[0] for d in diag_counts]

    # diag_sorted = diag_sorted[:10]

    # Perform value_counts and reset the index to make it a column
    temp_df = all_metadata['Tumor.group'].value_counts().reset_index()

    # Rename the columns to something meaningful
    temp_df.columns = ['Disease', 'Counts']

    # Now you can set 'Disease' as the index without an error
    temp_df = temp_df.set_index('Disease')

    # Convert to dictionary and proceed
    diag_counts = [(k, v['Counts']) for k, v in temp_df.to_dict('index').items()]
    label_names = [k for k, v in diag_counts]


    plt.rcParams.update({'font.size': 30})

    fig, ax = plt.subplots(1,1, figsize = (16,16))
    plt.grid(False)
    cm_display = ConfusionMatrixDisplay.from_predictions(results['query_class'], results['prediction'], labels=label_names,
                                            normalize=None, ax = ax, xticks_rotation = 'vertical', values_format = '.0f')
    cm_display.im_.colorbar.remove()  # Remove color bar
    plt.tight_layout()  # Apply tight layout
    # ax.tick_params(axis='x', rotation=45)  # Rotate x-axis tick labels
    # plt.setp(ax.get_xticklabels(), ha='right')  # Right-align the x-axis tick labels

    plt.savefig(f'{outpath}/confusion_matrix.png')

    cm = confusion_matrix(results['query_class'], results['prediction'], labels=label_names)

    # Save to CSV
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    cm_df.to_csv(f'{outpath}/confusion_matrix_raw.csv')

    # Create the confusion matrix
    cm = confusion_matrix(results['query_class'], results['prediction'], labels=label_names)

    # Calculate class-specific metrics from the confusion matrix
    class_metrics = []
    for i, class_label in enumerate(label_names):
        tp = cm[i, i]  # True Positives for the class
        fp = cm[:, i].sum() - tp  # False Positives for the class
        fn = cm[i, :].sum() - tp  # False Negatives for the class
        tn = cm.sum() - (tp + fp + fn)  # True Negatives for the class
        # print(class_label, tp, fp, fn, tn)

        # Accuracy for the specific class
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Precision for the specific class
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall for the specific class
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 score for the specific class
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Sensitivity (Recall) for the specific class
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Specificity for the specific class
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Balanced Accuracy for the specific class
        balanced_accuracy = (sensitivity + specificity) / 2

        # Append the metrics for this class
        class_metrics.append({
            'Disease': class_label,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Balanced Accuracy': balanced_accuracy
        })

    # Convert to a DataFrame
    df_class_metrics = pd.DataFrame(class_metrics)

    # Print or save the DataFrame to a CSV
    df_class_metrics.to_csv(f'{outpath}/disease_specific_metrics_updated.csv', index=False)

    plt.rcParams.update({'font.size': 16})

    # Create a barplot for F1 scores
    plt.figure(figsize=(10, 10))
    sns.barplot(x='Disease', y='F1', data=df_class_metrics, palette='viridis')

    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)

    # Add labels and title
    plt.xlabel('Disease')
    plt.ylabel('F1 Score')
    plt.title('F1 Score per Disease')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{outpath}/f1_score_barplot.png', dpi=300)
    # plt.show()

    plt.figure(figsize=(10, 10))
    sns.barplot(x='Disease', y='Balanced Accuracy', data=df_class_metrics, palette='viridis')
    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)
    # Add labels and title
    plt.xlabel('Disease')
    plt.ylabel('Balanced Accuracy')
    plt.title('Balanced Accuracy per Disease')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{outpath}/balanced_accuracy_barplot.png', dpi=300)

    plt.figure(figsize=(10, 10))
    sns.barplot(x='Disease', y='Recall', data=df_class_metrics, palette='viridis')
    # Set y-axis range from 0 to 1
    plt.ylim(0, 1)
    # Add labels and title
    plt.xlabel('Disease')
    plt.ylabel('Recall')
    plt.title('Recall per Disease')
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    # Show the plot
    plt.tight_layout()
    plt.savefig(f'{outpath}/recall_barplot.png', dpi=300)
    


    plt.rcParams.update({'font.size': 30})

    fig, ax = plt.subplots(1,1, figsize = (16,16))
    plt.grid(False)
    cm_display = ConfusionMatrixDisplay.from_predictions(results['query_class'], results['prediction'], labels=label_names,
                                            normalize=None, ax = ax, xticks_rotation = 'vertical', values_format = '.0f')
    cm_display.im_.colorbar.remove()  # Remove color bar
    plt.tight_layout()  # Apply tight layout
    plt.savefig(f'{outpath}/confusion_matrix_pred.png')

    plt.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(1,1, figsize = (16,16))
    plt.grid(False)
    cm_display = ConfusionMatrixDisplay.from_predictions(results['query_class'], results['prediction'], labels = label_names,
                                            normalize='true', ax = ax, xticks_rotation = 'vertical', values_format = '.2f')
    cm_display.im_.colorbar.remove()  # Remove color bar
    plt.title("Normalized by column ('recall')")
    plt.tight_layout()  # Apply tight layout
    plt.savefig(f'{outpath}/confusion_matrix_true.png')

    fig, ax = plt.subplots(1,1, figsize = (16,16))
    plt.grid(False)
    cm_display = ConfusionMatrixDisplay.from_predictions(results['query_class'], results['prediction'], labels = label_names,
                                            normalize='pred', ax = ax, xticks_rotation = 'vertical', values_format = '.2f')
    cm_display.im_.colorbar.remove()  # Remove color bar
    plt.title("Normalized by row ('precision')")
    plt.tight_layout()  # Apply tight layout
    plt.savefig(f'{outpath}/confusion_matrix_pred.png')
    cm_df = pd.DataFrame(cm_display.confusion_matrix, index=label_names, columns=label_names)
    # Save the DataFrame to a pickle file
    cm_df.to_pickle(f'{outpath}/confusion_matrix_pred.pkl')


    return sim_filtered, slides_filtered, all_metadata

def eval_old_model(results_in, sim_in, metadata_path, outpath):

    slides, bobs, mosaics, sjids, similarity_matrix = load_results(results_in, sim_in)
    all_metadata = load_metadata(metadata_path)
    slides, bobs, mosaics, sjids, similarity_matrix, indices = filter_data(slides, bobs, mosaics, sjids, similarity_matrix, all_metadata)

    results_summary, diag_sorted = make_nn_predictions(all_metadata, slides, similarity_matrix)

    acc = accuracy_score(results_summary['label'], results_summary['prediction'])
    f1 = f1_score(results_summary['label'], results_summary['prediction'], average = 'macro')
    pr = precision_score(results_summary['label'], results_summary['prediction'], average = 'macro')
    re = recall_score(results_summary['label'], results_summary['prediction'], average = 'macro')

    print('accuracy:', acc)
    print('f1:', f1)
    print('precision:', pr)
    print('recall:', re)

    fig, ax = plt.subplots(1,1, figsize = (16,16))
    plt.grid(False)
    ConfusionMatrixDisplay.from_predictions(results_summary['label'], results_summary['prediction'], 
                                            normalize='true', ax = ax, xticks_rotation = 'vertical', 
                                            labels = diag_sorted, values_format = '.2f')
    plt.savefig(os.path.join(outpath, './confusion_matrix_pred.png'))
    return similarity_matrix, slides, all_metadata


def main(model_name, agg_method, sim_matrix_path, metadata_path, outpath, remove_normal_slides, TOPK, tie_breaker_type):
    sim_in = sim_matrix_path
    metadata_path = metadata_path
    outpath = outpath
    print(outpath)
    os.makedirs(outpath, exist_ok = True)


    sim_filtered, slides_filtered, all_metadata = eval_new_model(sim_in, metadata_path, outpath, remove_normal_slides, TOPK, tie_breaker_type)

if __name__ == '__main__':


    args = parser.parse_args()
    model_name = args.model_name
    agg_method = args.agg_method
    TOPK = args.topk
    if args.remove_normal_slides:
        normal_dir_name = 'tumor_slides'
    else:
        normal_dir_name = 'all_slides'
    

    if args.tumor_tiles_only:
        tumor_dir_name = 'tumor_tiles_only'
    else:
        tumor_dir_name = 'all_tiles'

    outpath = f'/PROSPER_WORKSPACE/comet-feature-gen/comet-{model_name}/benchmark_final/{agg_method}/{normal_dir_name}/{tumor_dir_name}/top_{TOPK}/{args.tie_breaker_type}/'
    sim_matrix_path = f'/PROSPER_WORKSPACE/comet-feature-gen/comet-{model_name}/output/similarity_matrix/combined_similarity_matrix_{model_name}_{agg_method}_no_same_patient.pkl'
    # sim_matrix_path = f'/PROSPER_WORKSPACE/comet-feature-gen/comet-{model_name}/output/similarity_matrix/combined_similarity_matrix_{model_name}_{agg_method}.pkl'

    if args.tumor_tiles_only:
        sim_matrix_path = f'/PROSPER_WORKSPACE/comet-feature-gen/comet-{model_name}/output/similarity_matrix/combined_similarity_matrix_{model_name}_{agg_method}_tumor_only.pkl'
        model_name += '_tumor_only'
        
    metadata_path = '/PROSPER_WORKSPACE/comet-new-scans/data/final_slide_list/final_scan_id_disease_map.csv'
    main(model_name, agg_method, sim_matrix_path, metadata_path, outpath, args.remove_normal_slides, TOPK, args.tie_breaker_type)