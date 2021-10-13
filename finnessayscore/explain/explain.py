#!/usr/bin/env python3
import torch.nn

from captum.attr import visualization as viz
from captum.attr import IntegratedGradients, LayerConductance, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
import captum

import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer
from finnessayscore import data_reader, model
import pickle
import argparse

class Reshape(torch.nn.Module):

    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

# LayerIntegratedGradients only takes in inputs (tensor or tuple of tensors)
class ExplainWholeEssayClassModel(model.WholeEssayClassModel):
    def __init__(self, class_nums, class_weights=None, **config):
        super().__init__(class_nums, class_weights=class_weights, **config)

    #this really must have throughout as many inputs/outputs as what Captum sees
    def forward(self, batch_input_ids, batch_attention_mask, batch_token_type_ids):
        essay_enc = self.bert(input_ids=batch_input_ids,
                              attention_mask=batch_attention_mask,
                              token_type_ids=batch_token_type_ids) #BxS_LENxSIZE; BxSIZE
        #so the pooler out is now (fragments x 768)
        essay_enc = torch.sum(essay_enc.pooler_output, 0).expand((*essay_enc.pooler_output.shape))
        #so now we should have (fragments x 768) still, copied
        print("ESSAY ENC SHAPE",essay_enc.shape,essay_enc[:,:3])
        res= {name: layer(essay_enc) for name, layer in self.cls_layers.items()}
        return res
        
    def forward_old(self, batch_input_ids, batch_attention_mask, batch_token_type_ids):
        print("batch_input_ids", batch_input_ids)
        print("batch_attention_mask", batch_attention_mask)
        print("batch_token_type_ids", batch_token_type_ids)
        essay_enc = self.bert(input_ids=batch_input_ids.squeeze(0),
                              attention_mask=batch_attention_mask.squeeze(0),
                              token_type_ids=batch_token_type_ids.squeeze(0)) #BxS_LENxSIZE; BxSIZE
        essay_enc = torch.unsqueeze(torch.sum(essay_enc.pooler_output, 0), 0)
        res= {name: layer(essay_enc) for name, layer in self.cls_layers.items()}
        return res

    

def predict(input_ids, attention_mask, token_type_ids): #inputs, token_type_ids=None, position_ids=None, attention_mask=None):
    pred = trained_model(input_ids, attention_mask, token_type_ids)
    print("PREDICTION",pred["lab_grade"].shape)
    return pred["lab_grade"] #return the output of the classification layer

def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

def aggregate(inp,attrs,tokenizer):
    """detokenize and merge attributions"""
    detokenized=[]
    for l in inp[0][0].cpu().tolist():
        detokenized.append(tokenizer.convert_ids_to_tokens(l))
    print("ATTRS SHAPE",attrs.shape)
    attrs=attrs.cpu().tolist()
    
    #print("DETOKENIZED",detokenized)

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
    with open("delme_before_print", "wb") as f:
        pickle.dump([target,aggregated], f)
    
    to_print=""
    to_print = to_print+"<html><body>"
    x=captum.attr.visualization.format_word_importances([t for t,a in aggregated],[a for t,a in aggregated])
    to_print = to_print+"<b>"+str(target)+"</b>"
    to_print = to_print+"""<table style="border:solid;">"""+x+"</table>"
    to_print = to_print+"</body></html>"
    from IPython.core.display import HTML, display
    display(HTML(to_print))

def build_ref(essay_i, batch, tokenizer, device):
    """Given index and a batch, return reference
    input_ids, token_type_ids, and attention_mask"""
    ref_token_id = tokenizer.pad_token_id # A token used for generating token reference
    sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
    cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence

    # ref input token id
    ref_input_ids = torch.tensor([[token if token==cls_token_id or token==sep_token_id else ref_token_id for token in seg] for seg in batch["input_ids"][essay_i]])
    # ref_token_type_ids
    ref_token_type_ids = batch["token_type_ids"][essay_i]
    # ref_attention_mask
    ref_attention_mask = batch["attention_mask"][essay_i]
    
    print("ref_input",(torch.unsqueeze(ref_input_ids, dim=0),
            torch.unsqueeze(ref_token_type_ids, dim=0),
            torch.unsqueeze(ref_attention_mask, dim=0)))

    return (torch.unsqueeze(ref_input_ids, dim=0).to(device),
            torch.unsqueeze(ref_attention_mask, dim=0).to(device),
            torch.unsqueeze(ref_token_type_ids, dim=0).to(device))
            

def predict_and_explain(trained_model, tokenizer, obj_batch):
    trained_model.zero_grad() #to be safe perhaps it's not needed
    device=trained_model.device

    lig = LayerIntegratedGradients(predict, trained_model.bert.embeddings)
    print("obj_batch", obj_batch)
    assert len(obj_batch["input_ids"])==1 # the whole_essay model only takes 1 example at a time
#    prediction = predict(torch.nn.utils.rnn.pad_sequence(obj_batch["input_ids"],batch_first=True).to(device),
#                          torch.nn.utils.rnn.pad_sequence(obj_batch["attention_mask"],batch_first=True).to(device),
#                          torch.nn.utils.rnn.pad_sequence(obj_batch["token_type_ids"],batch_first=True).to(device),)
#    print("PREDICTIONS", prediction.shape)
    #for i, prediction in enumerate(predictions): # the whole_essay model only takes 1 example at a time, so index is 0
#    prediction_cls=int(torch.argmax(prediction))
#    print("Gold standard:", obj_batch["lab_grade"][0].shape)
#    print("Prediction:", ("1","2","3","4","5")[prediction_cls],"Weights:",prediction.tolist())
    ref_input = build_ref(0, obj_batch, tokenizer, device)
    #inp = (obj_batch["input_ids"][i].unsqueeze(0)[0].to(device),
    #       obj_batch["attention_mask"][i].unsqueeze(0)[0].to(device),
    #       obj_batch["token_type_ids"][i].unsqueeze(0)[0].to(device))
    inp = (torch.nn.utils.rnn.pad_sequence(obj_batch["input_ids"],batch_first=True).to(device),
           torch.nn.utils.rnn.pad_sequence(obj_batch["attention_mask"],batch_first=True).to(device),
           torch.nn.utils.rnn.pad_sequence(obj_batch["token_type_ids"],batch_first=True).to(device),)
#    all_tokens = [tokenizer.convert_ids_to_tokens(seg) for seg in inp[0][0]]
    #print("all_tokens", all_tokens)
    print("inp", inp[0].shape)
    print("attm",inp[1].shape)
    print("tt",inp[2].shape)
    print("ref_input",ref_input[0].shape,ref_input[1].shape,ref_input[2].shape)
    print("I AM HERE")
    inp=tuple([x.squeeze(0) for x in inp]) #I am actually squeezing the fake first dim out
    ref_input=tuple([x.squeeze(0) for x in ref_input])
    for target, classname in ((4,"5"),):
        print("INP SHAPE",inp[0].shape,inp[1].shape,inp[2].shape)
        attrs = lig.attribute(inputs=inp,
                                     baselines=ref_input,
                                     return_convergence_delta=False,
                                     target=target,
                                     internal_batch_size=1,n_steps=3)
        #try:
        #    with open("delme", "wb") as f:
        #        pickle.dump([obj_batch["essay"], attrs, delta], f)
        #    print("saved")
        #except Exception as e:
        #    print(e)

        print("ATTRS SHAPE",attrs.shape)
        print("ATTRS",attrs[:,:3,:1])
        sys.exit()
        attrs_sum = summarize_attributions(attrs)
        aggregated = aggregate(inp, attrs_sum, tokenizer)
        
        x=captum.attr.visualization.format_word_importances(all_tokens, attrs_sum)
        print("ATTRIBUTION WITH RESPECT TO",classname)
        print_aggregated(target, aggregated)
        #display(HTML(x))
        print()



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_label_smoothing', default=False, action="store_true", help="Use label smoothing")
    parser.add_argument('--smoothing', type=float, default=0, help="0: one-hot method, 0<x<1: smooth method")
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--run_id', help="Optional run id")

    pl.Trainer.add_argparse_args(parser)
    data_reader.JsonDataModule.add_argparse_args(parser)

    args = parser.parse_args()

    # paremeters
    checkpoint = "best_model.ckpt"

    # data
    data = data_reader.JsonDataModule.from_argparse_args(args)
    data.setup()

    # model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    trained_model = ExplainWholeEssayClassModel.load_from_checkpoint(checkpoint_path=checkpoint,
                                                                     class_nums=data.class_nums(),
                                                                     label_smoothing=args.use_label_smoothing)
    trained_model.eval()
    #trained_model.cuda() # needs around ~13GB of memory

    for batch in data.val_dataloader():
        predict_and_explain(trained_model, tokenizer, batch)
        break
