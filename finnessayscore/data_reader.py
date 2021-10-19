import json
import sys
import collections
from os.path import join as pjoin
from typing import Union, Dict, Any
import pickle
from .grade_scale import DEFAULT_GRADE_SCALE

import pytorch_lightning as pl
import transformers
import torch

from finnessayscore import preprocessing


class JsonDataModule(pl.LightningDataModule):

    batch_size: int = 0

    def __init__(self,
                 data_dir: str,
                 model_type: str,
                 batch_size: int = 20,
                 bert_model: str = "TurkuNLP/bert-base-finnish-cased-v1",
                 class_nums: Union[Dict[str, Any], str] = DEFAULT_GRADE_SCALE,
                 stride: int = 10,
                 max_length: int = 512,
                 **config):
        """Module for loading a pre-prepared directory containing a JSON file per split

        Args:
            data_dir: Directory containing pre-split dataset train.json, val.json and test.json
            model_type: Either trunc_essay, whole_essay, seg_essay, sentences, trunc_essay_ord, or pedantic_trunc_essay_ord
            class_nums: Pickle file with stored class_nums
            stride: Applies only when using whole_essay. The stride to use when feeding in multiple segments.
            max_length: Maximum length in BERT subword tokens when using whole_essay or seg_essay
        """
        super().__init__(self)
        assert model_type in ["whole_essay", "sentences", "trunc_essay", "trunc_essay_ord", "pedantic_trunc_essay_ord", "seg_essay"]
        self.data_dir = data_dir
        self.bert_model = bert_model
        self.batch_size = batch_size
        self.model_type = model_type
        if isinstance(class_nums, dict):
            self.class_nums_dict = class_nums
        else:
            with open(class_nums, "rb") as f:
                self.class_nums_dict = pickle.load(f)
            assert isinstance(self.class_nums_dict, dict)
        self.config = config

    def class_nums(self):
        return self.class_nums_dict

    def get_label_map(self):
        class_nums = self.class_nums()
        label_map = {}
        for k, lst in class_nums.items():
            label_map[k] = {i: l for i, l in enumerate(lst)}
        return label_map
        
    def prepare_data(self):
        pass

    def get_class_weights(self):
        print("Training set:")
        occurrences = self.basic_stats(self.train, get=True) # Dict[name]=List(class occurrences)
        print("Dev set:")
        self.basic_stats(self.val)

        # from occurrences to weights
        # https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights/
        weights_dict = {}
        for name, occ_lst in occurrences.items():
            weights = []
            total = sum(occ_lst)
            n_classes = len(occ_lst)
            for occ in occ_lst:
                weights.append(total/(n_classes*occ) if occ!=0 else 0)
            weights_dict[name] = torch.Tensor(weights)
        return weights_dict
    
    def basic_stats(self, data, get=False):
        counters = {name:collections.Counter() for name in self.class_nums()}
        for d in data:
            for k,v in d.items():
                if k.startswith("lab_"):
                    counters[k].update({v:1})
        for name, cntr in counters.items():
            print("OUTPUT:", name)
            total = sum(cnt for cls, cnt in cntr.items())
            for cls,cnt in cntr.most_common():
                cls = self.class_nums()[name][cls]
                print(f"{cls}   {cnt}/{total}={cnt/total*100:3.1f}")
            print()

        if get: # return the occurrence numbers
            occurrences = {}
            for name, cntr in counters.items():
                class_no = len(self.class_nums()[name])
                occ = []
                for i in range(class_no):
                    occ.append(cntr[i] if i in cntr else 0)
                occurrences[name] = occ
            return occurrences
        
    def break_essays(self, data):
        essays_processed = []
        for d in data:
            pieces = preprocessing.seg_by_char_index(d["essay"], 1500)
            for pcs in pieces:
                essays_processed.append({k:(v if k!="essay" else pcs) for k,v in d.items()})
        return essays_processed
    
    def tokenize_trunc_essay(self, data, tokenizer):
        # Tokenize and gather input ids, token type ids and attention masks which we need for the model
        tokenized = tokenizer([d["essay"] for d in data],
                              padding="longest",
                              truncation="longest_first",
                              max_length=512)
        for d,input_ids,token_type_ids,attention_mask in zip(data,tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]):
            d["input_ids"]=torch.LongTensor(input_ids)
            d["token_type_ids"]=torch.LongTensor(token_type_ids)
            d["attention_mask"]=torch.LongTensor(attention_mask)

    def tokenize_seg_essay(self, data, tokenizer):
        # Tokenize and gather input ids, token type ids and attention masks which we need for the model
        # One essay is cut into several segments, each of which becomes an example
        tokenized = tokenizer([d["essay"] for d in data],
                              padding="longest",
                              truncation="longest_first",
                              max_length=self.config["max_token"],
                              return_overflowing_tokens=True, return_offsets_mapping=True)
        new_data = []
        for i, old_i in enumerate(tokenized["overflow_to_sample_mapping"]):
            new_d = data[old_i].copy()
            new_d["input_ids"] = torch.LongTensor(tokenized["input_ids"][i])
            new_d["token_type_ids"] = torch.LongTensor(tokenized["token_type_ids"][i])
            new_d["attention_mask"] = torch.LongTensor(tokenized["attention_mask"][i])
            new_d["overflow_to_sample_mapping"] = tokenized["overflow_to_sample_mapping"][i]
            new_data.append(new_d)
        return new_data
            
    def tokenize_whole_essay(self, data, tokenizer):
        # Tokenize -> chunk -> add special token
        for d in data:
            tokenized = tokenizer(d["essay"], padding=True, truncation='longest_first', max_length=self.config["max_token"], return_overflowing_tokens=True, stride=self.config["stride"], return_offsets_mapping=True)
            #weights = sum([ for tokenized["attention_mask"]])
            #d["overflow_to_sample_mapping"] = tokenized["overflow_to_sample_mapping"]
            d["input_ids"] = torch.LongTensor(tokenized["input_ids"])
            d["token_type_ids"] = torch.LongTensor(tokenized["token_type_ids"])
            d["attention_mask"] = torch.LongTensor(tokenized["attention_mask"])

    def tokenize(self, data, tokenizer):
        # Tokenize and gather input ids, token type ids and attention masks which we need for the model
        for d in data:
            tokenized = tokenizer(d["sents"], padding="longest",truncation="longest_first", max_length=512)
            d["input_ids"] = torch.LongTensor(tokenized["input_ids"])
            d["token_type_ids"] = torch.LongTensor(tokenized["token_type_ids"])
            d["attention_mask"] = torch.LongTensor(tokenized["attention_mask"])

    #def tokenize_sbert(self, data, tokenizer):
        # Tokenize and gather input ids, token type ids and attention masks which we need for the model
        # https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    #    for d in data:
    #        tokenized = tokenizer(d["sents"], padding="longest",truncation="longest_first", max_length=512)
    #        d["input_ids"] = torch.LongTensor(tokenized["input_ids"])
    #        d["token_type_ids"] = torch.LongTensor(tokenized["token_type_ids"])
    #        d["attention_mask"] = torch.LongTensor(tokenized["attention_mask"])
                    
    def setup(self):
        # Read in from the JSONs
        for split in ["train", "val", "test"]:
            with open(pjoin(self.data_dir, f"{split}.json"), "r") as f:
                essay = json.load(f)
                # Classes into numerical indices
                for k, lst in self.class_nums().items():
                    essay = [d for d in essay if d[k]] # remove essays without labels
                    for d in essay:
                        d[k] = lst.index(d[k])
                setattr(self, split, essay)
        
        # essays are long, break them
        #self.train = self.break_essays(self.train)
        all_data = self.train + self.val + self.test
        print("After segmenting essays")
        self.basic_stats(all_data)
        
        # tokenization
        if self.model_type == "sentences":
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model,truncation=True)
            self.tokenize(all_data, tokenizer)
        elif self.model_type == "sbert":
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.bert_model) #, truncation=True)
            self.tokenize(all_data, tokenizer)
        elif "trunc_essay" in self.model_type:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model,truncation=True)
            # essays and prompts are in list, turn into string
            for d in all_data:
                d["essay"] = " ".join(d["essay"])
            self.tokenize_trunc_essay(all_data, tokenizer)
        elif self.model_type == "whole_essay":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model,truncation=True)
            for d in all_data:
                d["essay"] = " ".join(d["essay"])
            self.tokenize_whole_essay(all_data, tokenizer)
            # debug
            #all_data = [d for d in all_data if len(d['overflow_to_sample_mapping'])==1]
        elif self.model_type == "seg_essay":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model,truncation=True)
            for d in all_data:
                d["essay"] = " ".join(d["essay"])
            self.train = self.tokenize_seg_essay(self.train, tokenizer)
            self.val = self.tokenize_seg_essay(self.val, tokenizer)
            #self.tokenize_trunc_essay(self.val, tokenizer)

    def data_sizes(self):
        return len(self.train), len(self.val), len(self.test)
    
    def get_dataloader(self,which_set,**kwargs):
        """Just a utility so I don't need to repeat this in all the *_dataloader callbacks"""
        return torch.utils.data.DataLoader(which_set,
                                           collate_fn=collate_tensors_fn,
                                           batch_size=self.batch_size,
                                           **kwargs)
        
    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val)

    def test_dataloader(self):
        return self.get_dataloader(self.test)

def collate_tensors_fn(items):
    item=items[0] #this is a dictionary as it comes from the dataset

    pad_these=[] #let us be wannabe clever and pad everything which is a tensor
    tensor_these = [] # make a tensor of everything which is "lab_" key and an int
    list_these=[] #everything which is not a tensor we stick into a list
    
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            pad_these.append(k)
        elif isinstance(v, int) and k.startswith("lab_"):
            tensor_these.append(k)
        else:
            list_these.append(k)

    batch_dict={}
    for k in pad_these:
        #batch_dict[k]=torch.nn.utils.rnn.pad_sequence([item[k] for item in items],batch_first=True)
        batch_dict[k] = [item[k] for item in items]
    for k in tensor_these:
        batch_dict[k]=torch.LongTensor([item[k] for item in items])
    for k in list_these:
        batch_dict[k]=[item[k] for item in items]

    return batch_dict


def main_test():
    d=JsonDataModule([sys.stdin])
    d.setup()
    train=d.train_dataloader()
    for x in train:
        print(x)
    #tok=transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    #print(tok(["Minulla on koira","Minulla on kissa","Minulla on hauva"]))


if __name__=="__main__":
    main_test()
