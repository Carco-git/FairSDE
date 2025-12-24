
import torch.nn as nn
import torch
import numpy as np
import sys

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DualWriter:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        # This flush method is needed for compatibility with the standard stdout interface.
        self.terminal.flush()
        self.log.flush()
        
        
import torch.nn as nn
import pandas as pd
import torch
import sklearn.metrics as sklm
import numpy as np
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

def find_threshold(tol_output, tol_target):
    # to find this thresold, first we get the precision and recall without this, from there we calculate f1 score,
    # using f1score, we found this theresold which has best precsision and recall.  Then this threshold activation
    # are used to calculate our binary output.

    PRED_LABEL = ['disease']

    # create empty dfs
    thrs = []            
        
    for j in range(0, len(tol_output)):
        thisrow = {}
        truerow = {}

        # iterate over each entry in prediction vector; each corresponds to
        # individual label
        for k in range(len(PRED_LABEL)):
            thisrow["prob_" + PRED_LABEL[k]] = tol_output[j]
            truerow[PRED_LABEL[k]] = tol_target[j]
           
    for column in PRED_LABEL:
        
        thisrow = {}
        thisrow['label'] = column
        thisrow['bestthr'] = np.nan

        p, r, t = sklm.precision_recall_curve(tol_target, tol_output)
        # Choose the best threshold based on the highest F1 measure
        f1 = np.multiply(2, np.divide(np.multiply(p, r), np.add(r, p)))
        bestthr = t[np.where(f1 == max(f1))]
        thrs.append(bestthr)
        
        thisrow['bestthr'] = bestthr[0]

    return bestthr[0]


def calculate_auc(prediction, labels):
    fpr, tpr, thresholds = sklm.roc_curve(labels, prediction, pos_label=1)
    auc = sklm.auc(fpr, tpr)
    return auc

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize



def calculate_multiclass_auc(predictions, labels):
    if(predictions.shape[1]==2):
        predictions = predictions[:,1]
    auc_ovr = roc_auc_score(labels, predictions, average="macro", multi_class="ovr")
    
    return auc_ovr

def calculate_FPR_FNR(pred_df, test_meta, opt):
    
    dataset = opt['dataset_name']
    sens_attrs = opt['sensitive_name']
    if sens_attrs == 'Sex':
        sens = ['M', 'F']
        sens_attr_name = 'Sex'
    elif sens_attrs == 'Age':
        if opt['sens_classes'] == 2:
            sens_attr_name = 'Age_binary'
        elif opt['sens_classes'] == 5:
            sens_attr_name = 'Age_multi'
        elif opt['sens_classes'] == 4:
            sens_attr_name = 'Age_multi4'
        sens = np.arange(0, opt['sens_classes']).tolist()
    elif sens_attrs == 'Insurance':
        if opt['sens_classes'] == 2:
            sens_attr_name = 'Insurance_binary'
        elif opt['sens_classes'] == 5:
            sens_attr_name = 'Insurance'
        sens = np.arange(0, opt['sens_classes']).tolist()
        
    elif sens_attrs == 'Race':
        sens = ['White', 'non-White']
        sens_attr_name = 'Race'
    elif sens_attrs == 'skin_type':
        if opt['sens_classes'] == 2:
            sens_attr_name = 'skin_binary'
        elif opt['sens_classes'] == 6:
            sens_attr_name = 'skin_type'
        sens = np.arange(0, opt['sens_classes']).tolist()
    else:
        raise ValueError("{} not defined".format(sens_attrs))
    
    
    test_meta['index'] = test_meta.index
    pred_df = pred_df.merge(test_meta, left_on="index", right_on="index", suffixes=('', '_y'))
    sub_columns = ['index', 'pred', 'label'] + [sens_attr_name]
    pred_df = pred_df[sub_columns]
    
    FPR_y = []   
    FNR_y = []
    
    # calculate overall FPR, FNR
    overall_gt_fp, overall_gt_fn = pred_df[pred_df['label'] == 0], pred_df[pred_df['label'] == 1]
    overall_pred_fp = pred_df[(pred_df['pred'] == 1) & (pred_df['label'] == 0)]
    overall_pred_fn = pred_df[(pred_df['pred'] == 0) & (pred_df['label'] == 1)]
    overall_FPR, overall_FNR = len(overall_pred_fp) / len(overall_gt_fp), len(overall_pred_fn) / len(overall_gt_fn)
    for c in sens:
        
        # number of patient in subgroup with actual NF=0
        gt_fp = pred_df[(pred_df['label'] == 0) & (pred_df[sens_attr_name] == c)]
        # number of patient in subgroup with actual NF=1
        gt_fn = pred_df[(pred_df['label'] == 1) & (pred_df[sens_attr_name] == c)]
        
        pred_fp = pred_df[(pred_df['pred'] == 1) & (pred_df['label'] == 0) & (pred_df[sens_attr_name] == c)]
        pred_fn = pred_df[(pred_df['pred'] == 0) & (pred_df['label'] == 1) & (pred_df[sens_attr_name] == c)]
        
        if len(gt_fp) != 0 :
            FPR = len(pred_fp) / len(gt_fp)
            FPR_y.append(round(FPR, 4))
        else:
            FPR_y.append(np.NaN)
            
        if len(gt_fn) != 0 :
            FNR = len(pred_fn) / len(gt_fn)
            FNR_y.append(round(FNR, 4))
        else:
            FNR_y.append(np.NaN)
    return overall_FPR, overall_FNR, FPR_y, FNR_y


def conditional_AUC_binary(preds, labels, attrs):
    """
    Compute the conditional AUC of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: conditional AUC of A = 0, A = 1.
    """
    #print(preds.shape, labels.shape, attrs.shape)
    preds, labels, attrs = np.asarray(preds), np.asarray(labels), np.asarray(attrs)
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    idx = attrs == 0
    AUC_0 = calculate_auc(preds[idx], labels[idx])
    AUC_1 = calculate_auc(preds[~idx], labels[~idx])
    return AUC_0, AUC_1


def conditional_AUC_multi(preds, labels, attrs, sens_classes):
    """
    Compute the conditional AUC of A with multiple values (0, 1, 2, ...). All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 1, 2, n.
    """
    # print(preds.shape, labels.shape, attrs.shape)
    assert preds.shape[0] == labels.shape[0] and labels.shape[0] == attrs.shape[0]
    
    aucs = []
    for i in range(sens_classes):
        idx = attrs == i
        auc = calculate_multiclass_auc(preds[idx], labels[idx])
        aucs.append(auc)
    return aucs


def conditional_errors_binary(preds, labels, attrs):
    """
    Compute the conditional errors of A = 0/1. All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 0, error | A = 1.
    """
    
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean((preds == labels).astype('float'))
    idx = attrs == 0
    error_0 = 1 - np.mean((preds[idx] == labels[idx]).astype('float'))
    error_1 = 1 - np.mean((preds[~idx] == labels[~idx]).astype('float'))
    return cls_error, error_0, error_1


def conditional_errors_multi(preds, labels, attrs, sens_classes):
    """
    Compute the conditional errors of A with multiple values (0, 1, 2, ...). All the arguments need to be one-dimensional vectors.
    :param preds: The predicted label given by a model.
    :param labels: The groundtruth label.
    :param attrs: The label of sensitive attribute.
    :return: Overall classification error, error | A = 1, 2, n.
    """
    #print(preds.shape, labels.shape, attrs.shape)
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    cls_error = 1 - np.mean((preds == labels).astype('float'))
    
    errors = []
    for i in range(sens_classes):
        idx = attrs == i
        error = 1 - np.mean((preds[idx] == labels[idx]).astype('float'))
        errors.append(error.item())
    return cls_error, errors


def bce_loss(pred_probs, labels):
    bce = nn.BCELoss()
    
    pred_probs, labels = torch.from_numpy(pred_probs).flatten().cuda(), torch.from_numpy(labels).flatten().cuda()
    with torch.no_grad():
        loss = bce(pred_probs, labels)
    #print(loss)
    return loss.item()


def ce_loss(pred_probs, labels):
    ce = nn.CrossEntropyLoss()
    
    pred_probs, labels = torch.from_numpy(pred_probs).cuda(), torch.from_numpy(labels).cuda()
    with torch.no_grad():
        loss = ce(pred_probs, labels)
    #print(loss)
    return loss.item()


def expected_calibration_error(pred_probs, labels, num_bins=10, metric_variant="abs", quantile_bins=False):
    """
        Computes the calibration error with a binning estimator over equal sized bins
        See http://arxiv.org/abs/1706.04599 and https://arxiv.org/abs/1904.01685.
        Does not currently support sample weights
        https://github.com/MLforHealth/CXR_Fairness/blob/c2a0e884171d6418e28d59dca1ccfb80a3f125fe/cxr_fairness/metrics.py#L1557
    """
    if metric_variant == "abs":
        transform_func = np.abs
    elif (metric_variant == "squared") or (metric_variant == "rmse"):
        transform_func = np.square
    else:
        raise ValueError("provided metric_variant not supported")

    if quantile_bins:
        cut_fn = pd.qcut
    else:
        cut_fn = pd.cut

    bin_ids = cut_fn(pred_probs, num_bins, labels=False, retbins=False)
    df = pd.DataFrame({"pred_probs": pred_probs, "labels": labels, "bin_id": bin_ids})
    ece_df = (
        df.groupby("bin_id")
        .agg(
            pred_probs_mean=("pred_probs", "mean"),
            labels_mean=("labels", "mean"),
            bin_size=("pred_probs", "size"),
        )
        .assign(
            bin_weight=lambda x: x.bin_size / df.shape[0],
            err=lambda x: transform_func(x.pred_probs_mean - x.labels_mean),
        )
    )
    result = np.average(ece_df.err.values, weights=ece_df.bin_weight)
    if metric_variant == "rmse":
        result = np.sqrt(result)
    return result


def fnr_fpr_spe_sens_groups(preds, labels, attrs, sens_classes, specificity_val = 0.8, threshold = 0.5):
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    tpr_at_tnrs = []
    fnrs, fprs, recalls, specificitys = [], [], [], []
    eces, ces = [], []
    for i in range(sens_classes):
        idx = attrs == i
        tpr_at_tnrs.append(sensitivity_at_specificity(preds[idx], labels[idx], attrs[idx], specificity = specificity_val))
        fnrs.append(fnr_at_threshold(preds[idx], labels[idx], threshold = 0.5))
        fprs.append(fpr_at_threshold(preds[idx], labels[idx], threshold = 0.5))
        recalls.append(recall_at_threshold(preds[idx], labels[idx], threshold = 0.5))
        specificitys.append(specificity_at_threshold(preds[idx], labels[idx], threshold = 0.5))
        eces.append(expected_calibration_error(preds[idx], labels[idx]))
        ces.append(ce_loss(preds[idx], labels[idx]))
    return tpr_at_tnrs, fnrs, fprs, recalls, specificitys, eces, ces


def sensitivity_at_specificity(preds, labels, attrs, specificity):
    
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    fprs, tprs, thress = sklm.roc_curve(labels, preds)
    return interp1d(1 - fprs, tprs)(specificity)


def eqodd_at_specificity(preds, labels, attrs, specificity):
    
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    fprs, tprs, thress = sklm.roc_curve(labels, preds)
    thresh = interp1d(1 - fprs, thress)(specificity)
    
    return cal_eqodd(preds, labels, attrs, threshold = thresh)

def eqodd_at_sensitivity(preds, labels, attrs, sensitivity):
    
    assert preds.shape == labels.shape and labels.shape == attrs.shape
    fprs, tprs, thress = sklm.roc_curve(labels, preds)
    thresh = interp1d(tprs, thress)(sensitivity)
    
    return cal_eqodd(preds, labels, attrs, threshold = thresh)


def cal_eqodd(pred_probs, labels, attrs, threshold):
    tol_predicted = (pred_probs > threshold).astype('float')
    sens_idx = attrs == 0
    target_idx = labels == 0
    cls_error, error_0, error_1 = conditional_errors_binary(tol_predicted, labels, attrs)
    cond_00 = np.mean((tol_predicted[np.logical_and(sens_idx, target_idx)]))
    cond_10 = np.mean((tol_predicted[np.logical_and(~sens_idx, target_idx)]))
    cond_01 = np.mean((tol_predicted[np.logical_and(sens_idx, ~target_idx)]))
    cond_11 = np.mean((tol_predicted[np.logical_and(~sens_idx, ~target_idx)]))
    return (1 - 0.5 * (np.abs(cond_00 - cond_10) + np.abs(cond_01 - cond_11)))
    

def threshold_metric_fn(labels, pred_probs, sample_weight=None, threshold=0.5, metric_generator_fn=None):
    """
    Function that generates threshold metric functions.
    Calls a metric_generator_fn for customization
    """
    if metric_generator_fn is None:
        raise ValueError("metric_generator_fn must not be None")

    metric_fn = metric_generator_fn(
        threshold=threshold, 
    )
    if sample_weight is None:
        return metric_fn(pred_probs, labels )
    else:
        return metric_fn(pred_probs, labels, sample_weight=sample_weight)

    
def fpr_at_threshold(pred_probs, labels, sample_weight=None, threshold=0.5):
    """
    Computes specificity at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_fpr_at_threshold,
    )


def fnr_at_threshold(pred_probs, labels, sample_weight=None, threshold=0.5):
    """
    Computes specificity at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_fnr_at_threshold,
    )


def generate_fnr_at_threshold(threshold):
    return lambda pred_probs, labels : (1 - generate_recall_at_threshold(
                    threshold=threshold, 
                )(pred_probs, labels)
            )


def generate_fpr_at_threshold(threshold):

    return lambda pred_probs, labels : (1 - generate_specificity_at_threshold(
                    threshold=threshold, 
                )(pred_probs, labels)
            )
        
        
def recall_at_threshold(pred_probs, labels, sample_weight=None, threshold=0.5):
    """
    Computes recall at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_recall_at_threshold,
    )


def generate_recall_at_threshold(threshold, weighted=False, recalibrate=False):
    """
    Returns a lambda function that computes the recall at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    if not weighted:
        return lambda pred_probs, labels: sklm.recall_score(
            labels, 1.0 * (pred_probs >= threshold)
        )
    else:
        return lambda pred_probs, labels, sample_weight: sklm.recall_score(
            labels, 1.0 * (pred_probs >= threshold), sample_weight=sample_weight
        )


        
def specificity_at_threshold(pred_probs, labels, sample_weight=None, threshold=0.5):
    """
    Computes specificity at a threshold
    """
    return threshold_metric_fn(
        labels=labels,
        pred_probs=pred_probs,
        sample_weight=sample_weight,
        threshold=threshold,
        metric_generator_fn=generate_specificity_at_threshold,
    )


def generate_specificity_at_threshold(threshold, weighted=False):
    """
    Returns a lambda function that computes the specificity at a provided threshold.
    If weights = True, the lambda function takes a third argument for the sample weights
    """
    if not weighted:
        return (
            lambda pred_probs, labels: (
                (labels == 0) & (labels == (pred_probs >= threshold))
            ).sum()
            / (labels == 0).sum()
            if (labels == 0).sum() > 0
            else 0.0
        )
    else:
        return (
            lambda pred_probs, labels, sample_weight: (
                ((labels == 0) & (labels == (pred_probs >= threshold))) * sample_weight
            ).sum()
            / ((labels == 0) * sample_weight).sum()
            if (labels == 0).sum() > 0
            else 0.0
        )        
        

def get_worst_auc(log_dict):
    auc_dict = {}
    for key in log_dict:
        if 'auc-' in key:
            auc_dict[key] = log_dict[key]
    
    worst_auc = 1.
    group = -1
    for key, value in auc_dict.items():
        if value <= worst_auc:
            worst_auc = value
            group = key
    log_dict['worst_auc'] = worst_auc
    log_dict['worst_group'] = group
    return log_dict
    
def calculate_metrics(tol_output, tol_target, tol_sensitive, tol_index, sens_classes):

    correct = 0
    if( len(torch.tensor(tol_output).shape) != 1):
        tol_predicted = torch.tensor(tol_output).argmax(dim=1, keepdim=True)
    else:
        tol_predicted = ( torch.tensor(tol_output) > 0.5).int()
    correct += tol_predicted.eq( torch.tensor(tol_target).view_as(tol_predicted)).sum().item()
    
    
    acc = 100 * correct / len(tol_target)
    tol_predicted, tol_output, tol_target, tol_sensitive = np.asarray(tol_predicted), np.asarray(tol_output), \
    np.asarray(tol_target).squeeze(), np.asarray(tol_sensitive)


    auc = calculate_multiclass_auc(tol_output, tol_target)
    aucs = conditional_AUC_multi(tol_output, tol_target, tol_sensitive, sens_classes)
    
    log_dict = {"Overall AUC": auc,   
            }
    for i,  auc in enumerate( aucs):
        log_dict['auc-group_' + str(i)] = auc

        log_dict['Overall CE'] = ce_loss(tol_output, tol_target)

    log_dict = get_worst_auc(log_dict)
    return log_dict, tol_predicted, aucs


from datetime import datetime
import pytz

def get_et_time():
    eastern_time = datetime.now().astimezone(pytz.timezone('America/New_York'))
    formatted_time = eastern_time.strftime("%Y%m%d-%H%M%S")
    return formatted_time


from pulp import LpMaximize, LpProblem, LpVariable, lpSum, LpBinary
import pulp

import numpy as np
from scipy.optimize import minimize

def optimize_auc(tol_output, tol_output_erm, tol_target, tol_sensitive, tol_index, sensitive_attributes):

    log_dict, t_predictions, aucs_subgroup  = calculate_metrics(
        tol_output, tol_target, tol_sensitive, tol_index, sensitive_attributes)
    log_dict_erm, t_predictions_erm, aucs_subgroup_erm  = calculate_metrics(
        tol_output_erm, tol_target, tol_sensitive, tol_index, sensitive_attributes)

    max_gap_erm = max(aucs_subgroup_erm) - min(aucs_subgroup_erm)
    v = [LpVariable(f'v_{i}', cat=LpBinary) for i in range(sensitive_attributes)]
    model = LpProblem("Maximize Final AUC", LpMaximize)
    
    alpha_a = [(1 - v[i]) * aucs_subgroup[i] + v[i] * aucs_subgroup_erm[i] for i in range(sensitive_attributes)]

    max_alpha = LpVariable("max_alpha")
    min_alpha = LpVariable("min_alpha")

    for i in range(sensitive_attributes):
        model += max_alpha >= alpha_a[i]
        model += min_alpha <= alpha_a[i]

    model += max_alpha - min_alpha <= max_gap_erm

    for i in range(sensitive_attributes):
        model += alpha_a[i] >= aucs_subgroup_erm[i]

    model += lpSum(alpha_a)
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    use_erm = np.array([v[i].value() for i in range(sensitive_attributes)])
    mask = np.isin(tol_sensitive, np.where(use_erm == 1)[0])

    tol_output_final = np.where(mask[:, None], tol_output_erm, tol_output)
    log_dict_final, _, aucs_subgroup_adjust = calculate_metrics(tol_output_final, tol_target, tol_sensitive, tol_index, sensitive_attributes)
    overall_auc_final = log_dict_final['Overall AUC']

    return use_erm, overall_auc_final, log_dict_final, aucs_subgroup_adjust

def calculate_overall_auc(v_a, tol_output, tol_output_erm, tol_target, tol_sensitive, tol_index, sensitive_attributes):
    tol_output_final = np.where(v_a == 1, tol_output_erm, tol_output)
    log_dict_final, _, _ = calculate_metrics(tol_output_final, tol_target, tol_sensitive, tol_index, sensitive_attributes)
    return log_dict_final['Overall AUC']



import numpy as np
from scipy.optimize import minimize
def optimize_acc(total_accuracy, subgroup_acc, erm_accuracy, erm_accuracy_subgroups, total, args):
    n_groups = len(subgroup_acc)
    v_a_initial = np.zeros(n_groups)
    constraints = []
    for i in range(n_groups):
        constraints.append({
            'type': 'ineq',
            'fun': lambda v_a, i=i: (1 - v_a[i]) * subgroup_acc[i] + v_a[i] * erm_accuracy_subgroups[i] - erm_accuracy_subgroups[i]
        })
    max_diff_erm = np.max(erm_accuracy_subgroups) - np.min(erm_accuracy_subgroups)
    constraints.append({
        'type': 'ineq',
        'fun': lambda v_a: max_diff_erm - (np.max([(1 - v_a[i]) * subgroup_acc[i] + v_a[i] * erm_accuracy_subgroups[i] for i in range(n_groups)]) - 
                                            np.min([(1 - v_a[i]) * subgroup_acc[i] + v_a[i] * erm_accuracy_subgroups[i] for i in range(n_groups)]))
    })
    def objective_function(v_a):
        return -calculate_overall_acc(v_a, subgroup_acc, erm_accuracy_subgroups, total, n_groups)
    result = minimize(objective_function, v_a_initial, bounds=[(0, 1)] * n_groups, constraints=constraints, method='SLSQP')
    v_a_optimal = np.round(result.x).astype(int)
    overall_acc_final = calculate_overall_acc(v_a_optimal, subgroup_acc, erm_accuracy_subgroups, total, n_groups)
    return v_a_optimal, overall_acc_final

def calculate_overall_acc(v_a, subgroup_acc, erm_accuracy_subgroups, total, n_groups):

    subgroup_acc_final = [(1 - v_a[i]) * subgroup_acc[i] + v_a[i] * erm_accuracy_subgroups[i] for i in range(n_groups)]
    overall_acc_final = sum([subgroup_acc_final[i] * total[i] for i in range(n_groups)]) / sum(total)
    
    return overall_acc_final