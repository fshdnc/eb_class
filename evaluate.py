#!/usr/bin/env python3

import torch
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(conf_matrix, label_map, fname=None):
    import matplotlib.pyplot as plt
    import numpy
    import datetime
    
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

def evaluate(dataloader, model, label_map, plot_conf_mat=False):
    with torch.no_grad():
        preds = []
        target = []
        for batch in dataloader:
            needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids'] # some of the values cannot be put to gpu, filter those out
            # whole essay
            #output = model({k: v.cuda() for k, v in batch.items() if k in needed_for_prediction})
            # sentence
            output = model({k: [vv.cuda() for vv in v] for k, v in batch.items() if k in needed_for_prediction})
            preds.append(output) #["lab_grade"])
            target.append(batch["lab_grade"])
            #output = model_output_to_p(model({k: v.cuda() for k, v in batch.items()}))

    preds = [v for item in preds for k,vs in item.items() for v in vs]
    preds = [int(torch.argmax(pred)) for pred in preds]
    preds = [label_map["lab_grade"][p] for p in preds]
    #values, preds = torch.max(torch.tensor(preds), dim=1)
    target = [int(tt) for t in target for tt in t]
    target = [label_map["lab_grade"][p] for p in target]
    print("Predictions:", preds)
    print("Gold standard:", target)
    assert len(preds)==len(target)
    corrects = [1 if p==t else 0 for p, t in zip(preds,target)]
    print("Acc\t{}".format(sum(corrects)/len(corrects)))
    print("Predicted class number:",len(set(preds)))
    conf_mat = confusion_matrix(preds, target, labels=[l for i, l in label_map["lab_grade"].items()])
    print("Confusion matrix:\n", conf_mat)
    if plot_conf_mat:
        plot_confusion_matrix(conf_mat, label_map["lab_grade"], fname=None)

