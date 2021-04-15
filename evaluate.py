#!/usr/bin/env python3

import torch

def evaluate(dataloader, model, label_map):
    with torch.no_grad():
        preds = []
        for batch in dataloader:
            needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids'] # some of the values cannot be put to gpu, filter those out
            output = model({k: v.cuda() for k, v in batch.items() if k in needed_for_prediction})
            preds.append(output) #["lab_grade"])
            #output = model_output_to_p(model({k: v.cuda() for k, v in batch.items()}))

    preds = [v for item in preds for k,vs in item.items() for v in vs]
    preds = [int(torch.argmax(pred)) for pred in preds]
    #values, preds = torch.max(torch.tensor(preds), dim=1)
    target = [x['lab_grade'] for x in dataloader]
    target = [int(tt) for t in target for tt in t]
    print("Predictions:", [label_map["lab_grade"][p] for p in preds])
    print("Gold standard:", [label_map["lab_grade"][p] for p in target])
    assert len(preds)==len(target)
    corrects = [1 if p==t else 0 for p, t in zip(preds,target)]
    print("Acc\t{}".format(sum(corrects)/len(corrects)))
    print("Predicted class number:",len(set(preds)))

