#!/usr/bin/env python3

import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy
import datetime

def plot_confusion_matrix(conf_matrix, label_map, fname=None):
    norm_conf = []
    for i in conf_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            if float(a)!=0:
                tmp_arr.append(float(j)/float(a))
            else:
                tmp_arr.append(float(0))
        norm_conf.append(tmp_arr)
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(numpy.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = len(label_map), len(label_map)
    
    for x in range(width):
        for y in range(height):
            ax.annotate(str(conf_matrix[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    #label_list=[None]*len(label_map)
    #for label,number in label_map.items():
    #    label_list[number]=label
    label_list = [l for i, l in label_map.items()]
    plt.xticks(range(width), label_list)
    plt.yticks(range(height), label_list)
    plt.xlabel("Gold Standard")
    plt.ylabel("Prediction")
    if not fname:
        fname = str(datetime.datetime.now()).replace(":","").replace(" ","_")
    if not fname.endswith(".png"):
        fname = fname + ".png"
    plt.savefig(fname)

def _evaluate_seg_essay(dataloader, model, label_map):
    """
    Predictions for model type seg_essay, where an essay is chopped into several fragments, each used as an independent sample
    """
    with torch.no_grad():
        preds = []
        target = []
        overflow2sample_mapping = []
        for batch in dataloader:
            needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids', "overflow_to_sample_mapping"] # some of the values cannot be put to gpu, filter those out
            output = model({k: v for k, v in batch.items() if k in needed_for_prediction})
            preds.append(output) #["lab_grade"])
            target.append(batch["lab_grade"])
            overflow2sample_mapping.append(batch["overflow_to_sample_mapping"])
        # accuracy
        preds = [v for item in preds for k,vs in item.items() for v in vs]
        preds = [int(torch.argmax(pred)) for pred in preds]
        preds = [label_map["lab_grade"][p] for p in preds]
        #values, preds = torch.max(torch.tensor(preds), dim=1)
        target = [int(tt) for t in target for tt in t]
        target = [label_map["lab_grade"][p] for p in target]
        overflow2sample_mapping = [ii for i in overflow2sample_mapping for ii in i]

    # mapping [1,1,2,3,3,3] preds [3,4,3,5,5,4] -> preds [4,3,5]
    assert len(preds)==len(target) and len(preds)==len(overflow2sample_mapping)
    dict_pred = {}; dict_target = {}
    for p,t,i in zip(preds, target, overflow2sample_mapping):
        if i not in dict_pred:
            dict_pred[i] = [p]; dict_target[i] = [t]
        else:
            dict_pred[i].append(p); dict_target[i].append(t)
    for i in sorted(set([overflow2sample_mapping])): # maybe sort is not required, but let's put it here
        new_preds.append(round(numpy.mean(dict_pred[i])))
        new_target.append(round(numpy.mean(dict_target[i])))
    return numpy.array(new_preds), numpy.array(new_target)

def evaluate(dataloader, model, label_map, model_type, plot_conf_mat=False):
    if model_type == "seg_essay":
        preds, target = _evaluate_seg_essay(dataloader, model, label_map)
    else:
        with torch.no_grad():
            preds = []
            target = []
            for batch in dataloader:
                needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids', "overflow_to_sample_mapping"] # some of the values cannot be put to gpu, filter those out
                if model_type in ["trunc_essay"]:
                    output = model({k: v for k, v in batch.items() if k in needed_for_prediction})
                elif model_type in ["sentences", "whole_essay"]:
                    output = model({k: [vv.cuda() for vv in v] for k, v in batch.items() if k in needed_for_prediction})
                preds.append(output) #["lab_grade"])
                target.append(batch["lab_grade"])

        # accuracy
        preds = [v for item in preds for k,vs in item.items() for v in vs]
        preds = [int(torch.argmax(pred)) for pred in preds]
        preds = [label_map["lab_grade"][p] for p in preds]
        #values, preds = torch.max(torch.tensor(preds), dim=1)
        target = [int(tt) for t in target for tt in t]
        target = [label_map["lab_grade"][p] for p in target]

    #print("Predictions:", preds)
    #print("Gold standard:", target)
    assert len(preds)==len(target)
    corrects = [1 if p==t else 0 for p, t in zip(preds,target)]
    acc = sum(corrects)/len(corrects)
    #print("Acc\t{}".format(sum(corrects)/len(corrects)))

    # Pearson's correlation
    rho = numpy.corrcoef(numpy.array([int(p) for p in preds]), numpy.array([int(t) for t in target]))
    #print("Pearson's correlation:", rho[0][1])

    # Quadratic Weighted Kappa
    qwk = cohen_kappa_score(numpy.array([int(p) for p in preds]), numpy.array([int(t) for t in target]), weights='quadratic')

    # class number
    class_no = len(set(preds))
    #print("Predicted class number:",len(set(preds)))

    print("RESULTS\tacc\t{:.3f}\tpearson\t{:.3f}\tQWK\t{:.3f}\tclass_no\t{}".format(acc, rho[0][1], qwk, class_no))

    # confusion matrix
    conf_mat = confusion_matrix(preds, target, labels=[l for i, l in label_map["lab_grade"].items()])
    print("Confusion matrix:\n", conf_mat)
    if plot_conf_mat:
        plot_confusion_matrix(conf_mat, label_map["lab_grade"], fname=None)


