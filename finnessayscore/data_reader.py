import json
import sys
import collections
import random
random.seed(0)

import pytorch_lightning as pl
import transformers
import torch

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
            
    def tokenize_whole_essay(self, data, tokenizer):
        # Tokenize -> chunk -> add special token
        for d in data:
            tokenized = tokenizer(d["essay"], padding=True, truncation='longest_first', max_length=512, return_overflowing_tokens=True, stride=self.config["stride"], return_offsets_mapping=True)
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
        
        # tokenization
        if self.model_type=="sentences": #try:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_name,truncation=True)
            self.tokenize(self.all_data, tokenizer)
        elif self.model_type=="sbert":
            tokenizer = transformers.AutoTokenizer.from_pretrained(self.bert_model_name) #, truncation=True)
            self.tokenize(self.all_data, tokenizer)
            #self.tokenize_sbert(self.all_data, tokenizer)
        elif self.model_type=="trunc_essay": #except KeyError:
            tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_model_name,truncation=True)
            # essays and prompts are in list, turn into string
            for d in self.all_data:
                d["essay"] = " ".join(d["essay"])
            self.tokenize_trunc_essay(self.all_data, tokenizer)
        elif self.model_type == "whole_essay":
            tokenizer = transformers.BertTokenizerFast.from_pretrained(self.bert_model_name,truncation=True)
            for d in self.all_data:
                d["essay"] = " ".join(d["essay"])
            self.tokenize_whole_essay(self.all_data, tokenizer)
            # debug
            #self.all_data = [d for d in self.all_data if len(d['overflow_to_sample_mapping'])==1]
                
    def data_sizes(self):
        return len(self.train), len(self.dev), len(self.test)
    
    def get_dataloader(self,which_set,**kwargs):
        """Just a utility so I don't need to repeat this in all the *_dataloader callbacks"""
        return torch.utils.data.DataLoader(which_set,
                                           collate_fn=collate_tensors_fn,
                                           batch_size=self.batch_size,
                                           **kwargs)
        
    def train_dataloader(self):
        return self.get_dataloader(self.train, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.dev)

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
    

if __name__=="__main__":
    d=JsonDataModule([sys.stdin])
    d.setup()
    train=d.train_dataloader()
    for x in train:
        print(x)
    #tok=transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    #print(tok(["Minulla on koira","Minulla on kissa","Minulla on hauva"]))
