import csv
import sys
import random
random.seed(0)

import collections
import pytorch_lightning as pl
import transformers
import torch

def csv2dict(inp):
    r=csv.DictReader(inp,dialect="excel-tab")
    for d in r:
        new_d={"text":d["PYYNTO"], "lab_fat_pad":d["fat_pad"], "lab_fx":d["fx"]}
        yield new_d

class RowDataModule(pl.LightningDataModule):

    def __init__(self,fnames_or_files,batch_size=20,bert_model_name="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__(self)
        self.fnames=fnames_or_files
        self.bert_model_name=bert_model_name
        self.batch_size=batch_size

    def class_nums(self):
        return {"lab_fat_pad":["0","1"],"lab_fx":["0","1"]}
        
    def prepare_data(self):
        pass

    def print_basic_stats(self):
        counters={name:collections.Counter() for name in self.class_nums()}
        for d in self.all_data:
            for k,v in d.items():
                if k.startswith("lab_"):
                    counters[k].update({v:1})
        for name, cntr in counters.items():
            print("OUTPUT:", name)
            total=sum(cnt for cls,cnt in cntr.items())
            for cls,cnt in cntr.most_common():
                print(f"{cls}   {cnt}/{total}={cnt/total*100:3.1f}")
            print()
            
    
    def setup(self):
        # Read in from the CSV
        self.all_data=[]
        for fname in self.fnames:
            if isinstance(fname,str):
                if fname.endswith(".gz"):
                    f=gzip.open(fname,"rt")
                else:
                    f=open(fname,"rt")
            else:
                f=fname
            self.all_data.extend(list(csv2dict(f)))

            
        random.shuffle(self.all_data)
        self.print_basic_stats()
        # Tokenize and gather input ids, token type ids and attention masks which we need for the model
        tokenizer=transformers.BertTokenizer.from_pretrained(self.bert_model_name,truncation=True)
        tokenized=tokenizer([d["text"] for d in self.all_data])
        for d,input_ids,token_type_ids,attention_mask in zip(self.all_data,tokenized["input_ids"], tokenized["token_type_ids"], tokenized["attention_mask"]):
            d["input_ids"]=torch.LongTensor(input_ids)
            d["token_type_ids"]=torch.LongTensor(token_type_ids)
            d["attention_mask"]=torch.LongTensor(attention_mask)

        # Classes into numerical indices
        for k,lst in self.class_nums().items():
            for d in self.all_data:
                d[k]=lst.index(d[k])

        # Split to train-dev-test
        dev_start,test_start=int(len(self.all_data)*0.8),int(len(self.all_data)*0.9)
        self.train=self.all_data[:dev_start]
        self.dev=self.all_data[dev_start:test_start]
        self.test=self.all_data[test_start:]

    def data_sizes(self):
        return len(self.train), len(self.dev), len(self.test)
        
    def get_dataloader(self,which_set,**kwargs):
        """Just a utility so I don't need to repeat this in all the *_dataloader callbacks"""
        return torch.utils.data.DataLoader(which_set,collate_fn=collate_tensors_fn,batch_size=self.batch_size,**kwargs)
        
    def train_dataloader(self):
        return self.get_dataloader(self.train,shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.dev)

    def test_dataloader(self):
        return self.get_dataloader(self.test)

def collate_tensors_fn(items):
    item=items[0] #this is a dictionary as it comes from the dataset

    pad_these=[] #let us be wannabe clever and pad everything which is a tensor
    tensor_these=[] #make a tensor of everything which is a "lab_" key and an int
    list_these=[] #everything which is not a tensor we stick into a list
    
    for k,v in item.items():
        if isinstance(v,torch.Tensor):
            pad_these.append(k)
        elif isinstance(v,int) and k.startswith("lab_"):
            tensor_these.append(k)
        else:
            list_these.append(k)

    batch_dict={}
    for k in pad_these:
        batch_dict[k]=torch.nn.utils.rnn.pad_sequence([item[k] for item in items],batch_first=True)
    for k in tensor_these:
        batch_dict[k]=torch.LongTensor([item[k] for item in items])
    for k in list_these:
        batch_dict[k]=[item[k] for item in items]

    return batch_dict
    
if __name__=="__main__":
    d=RowDataModule([sys.stdin])
    d.setup()
    train=d.train_dataloader()
    for x in train:
        print(x)
    #tok=transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
    #print(tok(["Minulla on koira","Minulla on kissa","Minulla on hauva"]))
