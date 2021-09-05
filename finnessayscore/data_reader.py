import json
import sys
import collections
import random
random.seed(0)

import pytorch_lightning as pl
import transformers
import torch
from itertools import groupby
from transformers.file_utils import PaddingStrategy

from finnessayscore import preprocessing

class JsonDataModule(pl.LightningDataModule):

    def __init__(self, fnames_or_files, model_type, batch_size=20, bert_model_name="TurkuNLP/bert-base-finnish-cased-v1", **config):
        super().__init__(self)
        self.fnames = fnames_or_files
        self.bert_model_name = bert_model_name
        self.batch_size = batch_size
        self.model_type = model_type
        #self.label_map = {0:5, 1:1, 2:2, 3:3, 4:4}
        self.config = config

    def class_nums(self):
        return {"lab_grade": ["1","2","3","4","5"]}

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
        self.basic_stats(self.dev)

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
                    if isinstance(v, list):
                        for inner_v in v:
                            counters[k][inner_v] += 1
                    else:
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

    def remove_no_label(self):
        print("Removing data points without label(s)...")
        print("BEFORE", len(self.all_data))
        new = []
        for data in self.all_data:
            add = True
            for k in data.keys():
                if k.startswith("lab_"):
                    if not data[k]:
                        add = False
            if add:
                new.append(data)
        self.all_data = new
        print("AFTER", len(self.all_data))

    def clean_data(self):
        if self.model_type=="sentences":
            keep = ["essay", "lemma", "sents"]
        elif "essay" in self.model_type: # "whole_essay" or "trunc_essay"
            keep = ["essay"]
        for data in self.all_data:
            remove = []
            for k in data.keys():
                if k in keep or k.startswith("lab_"):
                    pass
                else:
                    remove.append(k)
            for k in remove:
                del data[k]
        
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
        def new_batch():
            return {
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
                "doc": [],
                "doc_in_batch": [],
                "lab_grade": [],
            }
        tokenized = tokenizer([d["essay"] for d in data],
                              padding=PaddingStrategy.MAX_LENGTH,
                              truncation="longest_first",
                              max_length=self.config["max_token"],
                              stride=self.config["stride"],
                              return_overflowing_tokens=True,
                              return_offsets_mapping=True)
        doc_to_segs = {}
        for k, v in enumerate(tokenized["overflow_to_sample_mapping"]):
            doc_to_segs.setdefault(v, []).append(k)
        data_batched = [new_batch()]
        doc_in_batch = 0
        for doc_idx, seg_idxs in doc_to_segs.items():
            if len(data_batched[-1]["input_ids"]) + len(seg_idxs) > self.batch_size:
                if len(seg_idxs) > self.batch_size:
                    raise RuntimeError(f"Batch size {self.batch_size} not big enough to fit document of size {len(seg_idxs)}: {data[doc_idx]!r}")
                data_batched.append(new_batch())
                doc_in_batch = 0
            for seg_idx in seg_idxs:
                for seg_key in ("input_ids", "token_type_ids", "attention_mask"):
                    data_batched[-1][seg_key].append(tokenized[seg_key][seg_idx])
                data_batched[-1]["doc"].append(doc_idx)
                data_batched[-1]["doc_in_batch"].append(doc_in_batch)
            for doc_key in ["lab_grade"]:
                data_batched[-1][doc_key].append(data[doc_idx][doc_key])
            doc_in_batch += 1
        for batch in data_batched:
            batch["num_docs"] = [len(batch["lab_grade"])]
        return data_batched

    def tokenize_whole_essay_nosegment(self, data, tokenizer):
        tokenized = tokenizer([d["essay"] for d in data],
                              padding=False,
                              truncation=False)
        for d,input_ids,token_type_ids,attention_mask in zip(data,tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]):
            d["input_ids"]=torch.LongTensor(input_ids)
            d["token_type_ids"]=torch.LongTensor(token_type_ids)
            d["attention_mask"]=torch.LongTensor(attention_mask)

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
        self.all_data=[]
        for fname in self.fnames:
            with open(fname, "r") as f:
                self.all_data.extend(json.load(f))

        # remove essays without labels
        self.remove_no_label()

        # remove key, value pairs that are not used
        self.clean_data()

        # Classes into numerical indices
        for k, lst in self.class_nums().items():
            for d in self.all_data:
                d[k] = lst.index(d[k])
        
        random.shuffle(self.all_data)
        self.basic_stats(self.all_data)

        # Split to train-dev-test
        dev_start,test_start=int(len(self.all_data)*0.8),int(len(self.all_data)*0.9)
        self.train=self.all_data[:dev_start]
        self.dev=self.all_data[dev_start:test_start]
        self.test=self.all_data[test_start:]

        # essays are long, break them
        #self.train = self.break_essays(self.train)
        self.all_data = self.train + self.dev + self.test
        print("After segmenting essays")
        self.basic_stats(self.all_data)

        def untokenize_essay():
            for d in self.all_data:
                d["essay"] = " ".join(d["essay"])
        
        # tokenization
        if self.model_type=="sentences": #try:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_name,truncation=True)
            self.tokenize(self.all_data, tokenizer)
        elif self.model_type=="sbert":
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.bert_model_name) #, truncation=True)
            self.tokenize(self.all_data, tokenizer)
            #self.tokenize_sbert(self.all_data, tokenizer)
        elif "trunc_essay" in self.model_type: #except KeyError:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_name,truncation=True)
            # essays and prompts are in list, turn into string
            untokenize_essay()
            self.tokenize_trunc_essay(self.all_data, tokenizer)
        elif self.model_type == "whole_essay":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model_name,truncation=True)
            untokenize_essay()
            self.train = self.tokenize_whole_essay(self.train, tokenizer)
            self.dev = self.tokenize_whole_essay(self.dev, tokenizer)
        elif self.model_type == "whole_essay_nosegment":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model_name,truncation=False)
            untokenize_essay()
            self.tokenize_whole_essay_nosegment(self.all_data, tokenizer)
        elif self.model_type=="seg_essay":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model_name,truncation=True)
            self.train = self.tokenize_seg_essay(self.train, tokenizer)
            self.dev = self.tokenize_seg_essay(self.dev, tokenizer)
            #self.tokenize_trunc_essay(self.dev, tokenizer)

    def data_sizes(self):
        return len(self.train), len(self.dev), len(self.test)
    
    def get_dataloader(self,which_set,**kwargs):
        """Just a utility so I don't need to repeat this in all the *_dataloader callbacks"""
        if self.model_type != "whole_essay":
            # Use manual batching for whole_essay
            kwargs["batch_size"] = self.batch_size
            kwargs["collate_fn"] = collate_tensors_fn
        else:
            kwargs["collate_fn"] = collate_tensors_whole_essay
        return torch.utils.data.DataLoader(which_set, **kwargs)
        
    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.dev)

    def test_dataloader(self):
        return self.get_dataloader(self.test)


def collate_tensors_whole_essay(items):
    assert len(items) == 1
    item = items[0]
    for k, v in item.items():
        if k in [
            "input_ids",
            "token_type_ids",
            "attention_mask",
            "doc",
            "lab_grade",
            "num_docs",
            "doc_in_batch",
        ]:
            item[k] = torch.LongTensor(v)

    return item


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
    

if __name__=="__main__":
    d=JsonDataModule([sys.stdin])
    d.setup()
    train=d.train_dataloader()
    for x in train:
        print(x)
    #tok=transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    #print(tok(["Minulla on koira","Minulla on kissa","Minulla on hauva"]))
