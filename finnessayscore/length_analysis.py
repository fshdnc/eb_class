#!/usr/bin/env python3

# investigating whether length affects the predicted score

import datetime
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from finnessayscore import data_reader, model
import pickle
import argparse

print("GPU availability:", torch.cuda.is_available())

def get_length_essay_as_batch(attention_mask_batch, overlap=10):
    """
    Calculate essay length in terms of number of tokens from the attention mask
    Input is a "batch" of batch_size 1, i.e. one essay per batch
    """
    assert len(attention_mask_batch)==1
    segs_attention_mask = attention_mask_batch[0]
    lengths = [sum(seg)-2 for seg in segs_attention_mask]
    essay_len = sum(lengths) - overlap*(len(lengths)-1)
    return essay_len

def _predict_seg_essay(dataloader, model, label_map):
    """
    Return the lengths and preds of each segment
    For the target, the segments are returned, not the essays
    """
    with torch.no_grad():
        preds = []
        target = []
        lengths = []
        for batch in dataloader:
            # getting the lengths of the examples in each batch
            batch_len = [int(sum(seg_m)-2) for seg_m in batch["attention_mask"]]
            lengths.extend(batch_len)

            needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids'] #, "overflow_to_sample_mapping"] # some of the values cannot be put to gpu, filter those out
            print("batch", batch)
            output = model({k: [vv.to(model.device) for vv in v] for k, v in batch.items() if k in needed_for_prediction})
            preds.extend([int(torch.argmax(pred)) for pred in output["lab_grade"]])
            target.extend([int(t) for t in batch["lab_grade"]])
    print("lengths, preds, target")
    print(lengths, preds, target, sep="\n")
    return lengths, preds, target

def predict(dataloader, model, label_map, model_type, overlap):
    if model_type == "seg_essay":
        lengths, preds, target = _predict_seg_essay(dataloader, model, label_map)
    else:
        with torch.no_grad():
            preds = []
            target = []
            lengths = []
            for batch in dataloader:
                # getting the lengths of the examples in each batch
                if "trunc_essay" in model_type:
                    batch_len = [sum(seg_m)-2 for seg_m in batch["attention_mask"]]
                    lengths.extend(batch_len)
                elif model_type=="whole_essay":
                    lengths.append(get_length_essay_as_batch(batch["attention_mask"], overlap=overlap))
                else:
                    raise NotImplementedError

                needed_for_prediction = ['input_ids', 'attention_mask', 'token_type_ids', "overflow_to_sample_mapping"] # some of the values cannot be put to gpu, filter those out
                if "trunc_essay" in model_type:
                    output = model({k: [vv.to(model.device) for vv in v] for k, v in batch.items() if k in needed_for_prediction})
                elif model_type in ["sentences", "whole_essay"]:
                    output = model({k: [vv.to(model.device) for vv in v] for k, v in batch.items() if k in needed_for_prediction})
                preds.append({
                    k: [int(i) for i in v]
                    for k, v
                    in model.out_to_cls(output).items()
                })
                target.append(batch["lab_grade"])

        # accuracy
        preds = [v for item in preds for k,vs in item.items() for v in vs]
        preds = [label_map["lab_grade"][p] for p in preds]
        #values, preds = torch.max(torch.tensor(preds), dim=1)
        target = [int(tt) for t in target for tt in t]
        target = [label_map["lab_grade"][p] for p in target]
        
    return preds, target, lengths

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', required=True, help="Trained model for analysis")
    parser.add_argument('--run_id', help="Optional run id")
    parser.add_argument('--use_label_smoothing', default=False, action="store_true", help="Use label smoothing")
    #parser.add_argument('--smoothing', type=float, default=0, help="0: one-hot method, 0<x<1: smooth method")

    args = parser.parse_args()
    assert args.model_type in ["whole_essay", "sentences", "trunc_essay", "trunc_essay_ord", "pedantic_trunc_essay_ord", "seg_essay"]


    if args.model_type=="sentences":
        for j in args.jsons:
            assert "parsed" in j
    if not args.run_id:
        args.run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")
    print("RUN_ID", args.run_id, sep="\t")


    # data
    pl.Trainer.add_argparse_args(parser)
    data_reader.JsonDataModule.add_argparse_args(parser)
    data.setup()


    # model
    if args.model_type=="whole_essay":
        m = model.WholeEssayClassModel
    elif args.model_type=="sentences":
        m = model.ClassModel
    elif args.model_type == "trunc_essay_ord":
        m = model.TruncEssayOrdModel
    elif args.model_type == "pedantic_trunc_essay_ord":
        m = model.PedanticTruncEssayOrdModel
    elif args.model_type in ["trunc_essay", "seg_essay"]:
        m = model.TruncEssayClassModel

    trained_model = m.load_from_checkpoint(checkpoint_path=args.load_checkpoint,
                                           bert_model=args.bert_path,
                                           class_nums=data.class_nums(),
                                           label_smoothing=args.use_label_smoothing)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    trained_model.eval()
    trained_model.cuda()

    import json
    preds, target, lengths = predict(data.val_dataloader(), trained_model, data.get_label_map(), model_type=args.model_type, overlap=args.whole_essay_overlap)
    
    with open("delme_lengths.pickle","wb") as f:
        pickle.dump([preds, target, lengths], f)


