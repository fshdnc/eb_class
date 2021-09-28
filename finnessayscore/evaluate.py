#!/usr/bin/env python3

import torch
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import numpy
import datetime
import seaborn
from itertools import islice


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()


def savefig_auto(plt, what, base, fname=None):
    if not fname:
        fname = base + "_" + str(datetime.datetime.now()).replace(":","").replace(" ","_")
    if not fname.endswith(".png"):
        fname = fname + ".png"
    print(what + " saved to", fname)
    plt.savefig(fname)


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
    savefig_auto(plt, "Confusion matrix", "confmat", fname)


def plot_beeswarm(outputs, targets, cutoffs, label_map, fname=None):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_xlabel('Gold Standard')
    ax.set_ylabel('Prediction')
    ax2 = ax.twinx()
    ax2.set_ylabel('Boundaries')
    labels = list(label_map.values())
    seaborn.swarmplot(
        x=targets,
        y=outputs,
        ax=ax,
        order=labels
    )
    for cutoff, label in zip(cutoffs.tolist(), labels[1:]):
        ax.axhline(y=cutoff)
        ax.annotate(">=" + label, (ax.get_xlim()[1], cutoff))
    savefig_auto(plt, "Beeswarm plot", "beeswarm", fname)

def plot_beeswarm_prob(x, probs, fname=None):
    # https://stackoverflow.com/questions/36153410/how-to-create-a-swarm-plot-with-matplotlib
    # x: ["correct", "correct", "wrong",...]
    # probs = [0.657, 0.872, 0.334, ...] # softmax probability for the predcited class
    fig = plt.figure()
    ax = fig.add_subplot(111)
    seaborn.set_style("whitegrid")
    ax = seaborn.swarmplot(x=x, y=probs)
    ax = seaborn.boxplot(x=x, y=probs,
                     showcaps=False,boxprops={'facecolor':'None'},
                     showfliers=False,whiskerprops={'linewidth':0})
    savefig_auto(plt, "Beeswarm plot probability", "probbee", fname)

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
        #values, preds = torch.max(torch.tensor(preds), dim=1)
        target = [int(tt) for t in target for tt in t]
        overflow2sample_mapping = [ii for i in overflow2sample_mapping for ii in i]

    # mapping [1,1,2,3,3,3] preds [3,4,3,5,5,4] -> preds [4,3,5]
    assert len(preds)==len(target) and len(preds)==len(overflow2sample_mapping)
    dict_pred = {}; dict_target = {}
    for p,t,i in zip(preds, target, overflow2sample_mapping):
        if i not in dict_pred:
            dict_pred[i] = [p]; dict_target[i] = [t]
        else:
            dict_pred[i].append(p); dict_target[i].append(t)
    new_preds = []; new_target = []
    for i in sorted(set(overflow2sample_mapping)): # maybe sort is not required, but let's put it here
        new_preds.append(round(numpy.mean(dict_pred[i])))
        new_target.append(round(numpy.mean(dict_target[i])))
    new_preds = [label_map["lab_grade"][p] for p in new_preds]
    new_target = [label_map["lab_grade"][p] for p in new_target]
    return new_preds, new_target


def evaluate(dataloader, model, label_map, model_type, plot_conf_mat=False, do_plot_beeswarm=False, do_plot_prob=False, fname=None):
    if model_type == "seg_essay":
        preds, target = _evaluate_seg_essay(dataloader, model, label_map)
    else:
        outputs = []
        preds = []
        target = []
        all_outs = [] # the neural network output for all the classes
        with torch.no_grad():
            for batch in dataloader:
                needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids', "overflow_to_sample_mapping"] # some of the values cannot be put to gpu, filter those out
                if model_type.endswith("_ord"):
                    forward = model.forward_score
                    to_cls = model.score_to_cls
                else:
                    forward = model
                    to_cls = model.out_to_cls
                output = forward({k: [vv.to(model.device) for vv in v] for k, v in batch.items() if k in needed_for_prediction})
                all_outs.append(output["lab_grade"].tolist()) #print(output)
                out = output["lab_grade"][:, 0].tolist()
                outputs.extend(out)
                preds.append({
                    k: [int(i) for i in v]
                    for k, v
                    in to_cls(output).items()
                })
                target.append(batch["lab_grade"])

        # accuracy
        preds = [v for item in preds for k,vs in item.items() for v in vs]
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
        plot_confusion_matrix(conf_mat, label_map["lab_grade"], fname="confmat_"+fname if fname else None)

    if do_plot_beeswarm:
        plot_beeswarm(outputs, target, model.cutoffs_score_scale()["lab_grade"], label_map["lab_grade"], fname="beeswarm_"+fname if fname else None)

    if do_plot_prob:
        probs = []
        for x in all_outs: # list of list of list, second list batch size, third logits
            for xx in x:
                probs.append(max(softmax(xx)))
        x = []
        for pred, tar in zip(preds,target):
            if pred==tar:
                x.append("correct")
            else:
                x.append("wrong")
        #print("x", x, len(x))
        #print("probs", probs, len(probs))
        plot_beeswarm_prob(x, probs, fname="probbee_"+fname if fname else None)


