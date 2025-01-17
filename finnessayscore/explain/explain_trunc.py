#!/usr/bin/env python3

# explainability for trunc_essay model
from captum.attr import LayerIntegratedGradients
import captum

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from finnessayscore import data_reader, model
import pickle
import argparse
import datetime
from .utils import ControlTokenFilter, UposFilter, NerFilter, UnionFilter

print("GPU availability:", torch.cuda.is_available())

BERT_MAX_SEQUENCE_LENGTH = 512


# LayerIntegratedGradients only takes in inputs (tensor or tuple of tensors)
class ExplainTruncEssayClassModel(model.TruncEssayClassModel):
    def __init__(self, class_nums, bert_model, class_weights=None, **config):
        super().__init__(class_nums, bert_model, class_weights, **config)
    def forward(self, batch_input_ids, batch_attention_mask, batch_token_type_ids):
        enc = self.bert(input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        token_type_ids=batch_token_type_ids) #BxS_LENxSIZE; BxSIZE
        if "pooling" in self.config:
            if self.config["pooling"]=="mean":
                return {name: layer(torch.mean(enc.last_hidden_state, axis=1)) for name, layer in self.final_layers.items()}
            elif self.config["pooling"]=="cls":
                return {name: layer(enc.pooler_output) for name, layer in self.final_layers.items()}
            else:
                raise ValueError("Pooling method specified not known.")
        else: #defaults to cls
            return {name: layer(enc.pooler_output) for name, layer in self.final_layers.items()}


class ForwardAsForwardScoreMixin:
    def forward(
        self,
        batch_input_ids,
        batch_attention_mask,
        batch_token_type_ids
    ):
        enc = self.bert(input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                        token_type_ids=batch_token_type_ids)
        return self._score_out(enc)


class ExplainTruncEssayOrdModel(
    ForwardAsForwardScoreMixin,
    model.TruncEssayOrdModel
):
    pass


class ExplainPedanticTruncEssayOrdModel(
    ForwardAsForwardScoreMixin,
    model.PedanticTruncEssayOrdModel
):
    pass


# helper functions
def predict(input_ids, attention_mask, token_type_ids): #inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    pred = trained_model(input_ids, attention_mask, token_type_ids)
    return pred["lab_grade"] #return the output of the classification layer

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def aggregate(inp,attrs,tokenizer):
    """detokenize and merge attributions"""
    detokenized=[]
    for l in inp[0].cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    attrs=attrs.cpu().tolist()

    aggregated=[]
    for token,a_val in zip(detokenized[0],attrs): #One text from the batch at a time!
        if token.startswith("##"):
            #This is a continuation. We need to pool by absolute value, i.e. pick the most extreme one
            current_tok,current_a_val=aggregated[-1] #this is what we have so far
            if abs(current_a_val)>abs(a_val): #what we have has larger absval
                aggregated[-1]=(aggregated[-1][0]+token[2:],aggregated[-1][1])
            else:
                aggregated[-1]=(aggregated[-1][0]+token[2:],a_val) #the new value had a large absval, let's use that
        else:
            aggregated.append((token,a_val))
    return aggregated

def print_aggregated(target,aggregated):
    
    to_print=""
    to_print = to_print+"<html><body>"
    x=captum.attr.visualization.format_word_importances([t for t,a in aggregated],[a for t,a in aggregated])
    to_print = to_print+"<b>"+str(target)+"</b>"
    to_print = to_print+"""<table style="border:solid;">"""+x+"</table>"
    to_print = to_print+"</body></html>"
    from IPython.core.display import HTML, display
    display(HTML(to_print))

def build_ref(essay_i, batch, token_filter, tokenizer, device):
    """Given index and a batch, return reference
    input_ids, token_type_ids, and attention_mask"""
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    
    token_filter.update_essay(essay_i, batch)

    # ref input token id
    tok_idx = 0
    seg_filtered = []
    for token in batch["input_ids"][essay_i]:
        ignore_token = token_filter(tok_idx, token)
        if not ignore_token:
            token = ref_token_id
        seg_filtered.append(token)
        tok_idx += 1
    ref_input_ids = torch.tensor(seg_filtered)
    # ref_token_type_ids
    ref_token_type_ids = batch["token_type_ids"][essay_i]
    # ref_attention_mask
    ref_attention_mask = batch["attention_mask"][essay_i]
    
    return (torch.unsqueeze(ref_input_ids, dim=0).to(device),
            torch.unsqueeze(ref_attention_mask, dim=0).to(device),
            torch.unsqueeze(ref_token_type_ids, dim=0).to(device))


def get_predictions(obj_batch, device):
    return predict(torch.nn.utils.rnn.pad_sequence(obj_batch["input_ids"],batch_first=True).to(device),
                   torch.nn.utils.rnn.pad_sequence(obj_batch["attention_mask"],batch_first=True).to(device),
                   torch.nn.utils.rnn.pad_sequence(obj_batch["token_type_ids"],batch_first=True).to(device))


def predict_and_explain(trained_model, tokenizer, obj_batch, token_filter, target_layer_func=lambda x: x.bert.embeddings, labels=("1","2","3","4","5")):
    device = trained_model.device
    trained_model.zero_grad() #to be safe perhaps it's not needed

    lig = LayerIntegratedGradients(predict, target_layer_func(trained_model)) #trained_model.bert.embeddings)
    predictions = get_predictions(obj_batch, device)

    aggregate_batch = []
    for i, prediction in enumerate(predictions):
        aggregate_essay = []
        prediction_cls=int(torch.argmax(prediction))
        print("Gold standard:", obj_batch["lab_grade"][i])
        print("Prediction:", labels[prediction_cls],"Weights:",prediction.tolist()) # default ("1","2","3","4","5")
        ref_input = build_ref(i, obj_batch, token_filter, tokenizer, device)
        inp = (obj_batch["input_ids"][i].unsqueeze(0).to(device),
               obj_batch["attention_mask"][i].unsqueeze(0).to(device),
               obj_batch["token_type_ids"][i].unsqueeze(0).to(device))
        for target, classname in enumerate(labels): # default ("1","2","3","4","5")
            attrs, delta = lig.attribute(inputs=inp,
                                         baselines=ref_input,
                                         return_convergence_delta=True,
                                         target=target,
                                         internal_batch_size=1)
            attrs_sum = summarize_attributions(attrs)
            aggregated = aggregate(inp, attrs_sum, tokenizer)
            aggregate_essay.append(aggregated)
            #x=captum.attr.visualization.format_word_importances(all_tokens,attrs_sum)
            #print("ATTRIBUTION WITH RESPECT TO",classname)
            #print_aggregated(target, aggregated)
            #display(HTML(x))
            #print()
        aggregate_batch.append((obj_batch["lab_grade"][i].tolist(), labels[prediction_cls], aggregate_essay))
    return aggregate_batch


def predict_and_explain_ord(trained_model, tokenizer, obj_batch, token_filter, target_layer_func=lambda x: x.bert.embeddings, labels=("1","2","3","4","5")):
    device = trained_model.device
    ligs = [
        LayerIntegratedGradients(predict, target_layer_func(trained_model)),
        LayerIntegratedGradients(lambda *args, **kwargs: -predict(*args, **kwargs), target_layer_func(trained_model)),
    ]
    predictions = get_predictions(obj_batch, device)
    aggregate_batch = []
    for i, prediction in enumerate(predictions):
        aggregate_essay = []
        print("Gold standard:", obj_batch["lab_grade"][i])
        print("Prediction:", prediction)
        ref_input = build_ref(i, obj_batch, token_filter, tokenizer, device)
        inp = (obj_batch["input_ids"][i].unsqueeze(0).to(device),
               obj_batch["attention_mask"][i].unsqueeze(0).to(device),
               obj_batch["token_type_ids"][i].unsqueeze(0).to(device))
        for lig in ligs:
            attrs, delta = lig.attribute(inputs=inp,
                                         baselines=ref_input,
                                         return_convergence_delta=True,
                                         internal_batch_size=1)
            attrs_sum = summarize_attributions(attrs)
            aggregated = aggregate(inp, attrs_sum, tokenizer)
            aggregate_essay.append(aggregated)
        aggregate_batch.append((obj_batch["lab_grade"][i].tolist(), prediction.tolist()[0], aggregate_essay))
    return aggregate_batch


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=False, action="store_true", help="Use GPU")
    parser.add_argument('--load_checkpoint', required=True, help="Trained model for explainability analysis")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_label_smoothing', default=False, action="store_true", help="Use label smoothing")
    parser.add_argument('--smoothing', type=float, default=0, help="0: one-hot method, 0<x<1: smooth method")
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--run_id', help="Optional run id")
    parser.add_argument('--pooling', default="cls", help="only implemented for trunc_essay model, cls or mean")
    parser.add_argument('--exclude_upos', nargs="*", help="exclude tokens with these upos by adding them to empty")
    parser.add_argument('--exclude_ner', nargs="*", help="exclude tokens with these ner tags by adding them to empty. Can specify ALL to exclude all named entities.")

    pl.Trainer.add_argparse_args(parser)
    data_reader.JsonDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    #args, unknown = parser.parse_known_args()
    #print(args); print(unknown); exit()

    if not args.run_id:
        args.run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")
    print("RUN_ID", args.run_id, sep="\t")
    if args.class_nums:
        with open(args.class_nums, "rb") as f:
            args.class_nums = pickle.load(f)
    print(args)

    # data
    data = data_reader.JsonDataModule.from_argparse_args(args)
    data.setup()
    class_weights = data.get_class_weights()

    # model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    if args.model_type == "pedantic_trunc_essay_ord":
        model_cls = ExplainPedanticTruncEssayOrdModel
        predict_and_explain_func = predict_and_explain_ord
    elif args.model_type == "trunc_essay_ord":
        model_cls = ExplainTruncEssayOrdModel
        predict_and_explain_func = predict_and_explain_ord
    else:
        model_cls = ExplainTruncEssayClassModel
        predict_and_explain_func = predict_and_explain
    token_filter = ControlTokenFilter(tokenizer)
    if args.exclude_upos:
        token_filter = UnionFilter(token_filter, UposFilter(tokenizer, args.exclude_upos))
    if args.exclude_ner:
        token_filter = UnionFilter(token_filter, NerFilter(tokenizer, args.exclude_ner))
    trained_model = model_cls.load_from_checkpoint(checkpoint_path=args.load_checkpoint,
                                                   bert_model=args.bert_model,
                                                   class_nums=data.class_nums(),
                                                   label_smoothing=args.use_label_smoothing,
                                                   pooling=args.pooling,
                                                   class_weights=class_weights)

    trained_model.eval()
    if args.gpu:
        trained_model.cuda()
    aggregates = [] # list of (gold_standard, [attributions for 1-5])
    #count = 0
    import json

    # embedding layer
    agg_layer = []
    count = 0
    t = lambda x: x.bert.encoder.layer[l]
    for batch in data.val_dataloader():
        if count<30:
            agg_batch = predict_and_explain_func(
                trained_model, tokenizer, batch, token_filter,
                labels=args.class_nums["lab_grade"] if args.class_nums else ("1","2","3","4","5"))
            agg_layer.extend(agg_batch)
            count += 1
    aggregates.append(agg_layer)

    for l in range(11):
        agg_layer = []
        count = 0
        t = lambda x: x.bert.encoder.layer[l]
        for batch in data.val_dataloader():
            if count<30:
                agg_batch = predict_and_explain_func(
                    trained_model, tokenizer, batch, token_filter, target_layer_func=t,
                    labels=args.class_nums["lab_grade"] if args.class_nums else ("1","2","3","4","5"))
                agg_layer.extend(agg_batch)
            count += 1
        aggregates.append(agg_layer)
        #if count%1==0: # save attributions every two batches
        with open(args.run_id+".json","wt") as f:
            json.dump(aggregates, f)
            #print("Save predictions for {0} essays".format(len(aggregates)))
            print("Save predictions for {0} layers".format(len(aggregates)))

