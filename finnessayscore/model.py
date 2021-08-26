import pytorch_lightning as pl
import torch.nn.functional as F
import transformers
import torch
import torchmetrics
import gc
from finnessayscore.loss import LabelSmoothingLoss

class AbstractModel(pl.LightningModule):

    def __init__(self, class_nums, class_weights=None, **config):
        """
        Abstract base class for other training modules, defines:
            training_step
            validation_step
            validation_epoch_end
            configure_optimizer
        class_weights: Dict[name]=torch.Tesnor([weights])
        """
        super().__init__()
        self.train_acc = torch.nn.ModuleDict({name: torchmetrics.Accuracy() for name in class_nums})
        self.val_acc = torch.nn.ModuleDict({name: torchmetrics.Accuracy() for name in class_nums})
        self.train_qwk = torch.nn.ModuleDict({name: torchmetrics.CohenKappa(num_classes=len(lst), weights='quadratic') for name, lst in class_nums.items()})
        self.val_qwk = torch.nn.ModuleDict({name: torchmetrics.CohenKappa(num_classes=len(lst), weights='quadratic') for name, lst in class_nums.items()})
        if class_weights is None:
            self.class_weights = {name: None for name in class_nums}
        else:
            self.class_weights = class_weights
        self.config = config

    def training_step(self,batch,batch_idx):
        out = self(batch)
        losses = []
        pbar = {}
        for name in self.cls_layers:
            if self.config["label_smoothing"]:
                loss = self.losses[name](out[name], batch[name])
            else:
                loss = F.cross_entropy(out[name], batch[name], weight=self.class_weights[name])
            losses.append(loss)
            acc = self.train_acc[name](out[name], batch[name])
            self.log(f'train_acc_{name}', acc*100)
            self.log(f'train_loss_{name}', loss)
            pbar["acc_"+name] = f"{acc*100:03.1f}"
            qwk = self.train_qwk[name](out[name], batch[name])
            self.log(f'train_qwk_{name}', qwk)
        return {"loss":sum(losses), "progress_bar":pbar}

    def training_epoch_end(self, _):
        for name in self.cls_layers:
            print(f"train_acc_{name}", self.train_acc[name].compute()*100)
            self.log(f"train_acc_{name}", self.train_acc[name].compute()*100)
            self.train_acc[name].reset()
            print(f"train_qwk_{name}", self.train_qwk[name].compute()*100)
            self.log(f"train_qwk_{name}", self.train_qwk[name].compute()*100)
            self.train_qwk[name].reset()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        #losses = []
        for name in self.cls_layers:
            if self.config["label_smoothing"]:
                loss = self.losses[name](out[name], batch[name])
            else:
                loss = F.cross_entropy(out[name], batch[name], weight=self.class_weights[name])
            #losses.append(loss)
            self.log(f'train_loss_{name}', loss)
            self.val_acc[name](out[name], batch[name])
            self.val_qwk[name](out[name], batch[name])

    def validation_epoch_end(self, _):
        for name in self.cls_layers:
            print(f"val_acc_{name}", self.val_acc[name].compute()*100)
            self.log(f"val_acc_{name}", self.val_acc[name].compute()*100)
            self.val_acc[name].reset()
            print(f"val_qwk_{name}", self.val_qwk[name].compute()*100)
            self.log(f"val_qwk_{name}", self.val_qwk[name].compute()*100)
            self.val_qwk[name].reset()
            
    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(),
                                                    lr=self.config["lr"])
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                              num_warmup_steps=int(self.config["num_training_steps"]*0.1),
                                                                              num_training_steps=self.config["num_training_steps"])
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


class ClassModel(AbstractModel):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1", class_weights=None, **config):
        """
        an essay is represented by average bert encodings of each sentence
        class_weights: Dict[name]=torch.Tesnor([weights])
        """
        super().__init__(class_nums, class_weights, **config)
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        if self.config["label_smoothing"]:
            self.losses = {name: LabelSmoothingLoss(len(lst), smoothing=self.config["smoothing"], weight=self.class_weights[name]) for name, lst in class_nums.items()}

    def forward(self, batch):
        first = True
        enc = None
        for i in range(len(batch["input_ids"])):
            gc.collect()
            sample_enc = self.bert(input_ids=batch['input_ids'][i],
                                   attention_mask=batch['attention_mask'][i],
                                   token_type_ids=batch['token_type_ids'][i]) #BxS_LENxSIZE; BxSIZE
            sample_enc = torch.unsqueeze(torch.mean(sample_enc.pooler_output, 0), 0)
            if first:
                enc = sample_enc
                first = False
            else:
                enc = torch.cat((enc, sample_enc), dim=0)
        return {name: layer(enc) for name, layer in self.cls_layers.items()}


class WholeEssayClassModel(AbstractModel):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1", class_weights=None, **config):
        """
        An essay is represented by average bert encodings of its chunks
        class_weights: Dict[name]=torch.Tesnor([weights])
        """
        super().__init__(class_nums, class_weights, **config)
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        if self.config["label_smoothing"]:
            self.losses = {name: LabelSmoothingLoss(len(lst), smoothing=self.config["smoothing"], weight=self.class_weights[name]) for name, lst in class_nums.items()}

    def forward(self, batch):
        # one essay per batch, i.e. batch_size 1
        essay_enc = self.bert(input_ids=batch['input_ids'][0],
                              attention_mask=batch['attention_mask'][0],
                              token_type_ids=batch['token_type_ids'][0])
        essay_enc = torch.unsqueeze(torch.sum(essay_enc.pooler_output, 0), 0)
        return {name: layer(essay_enc) for name, layer in self.cls_layers.items()}


class TruncEssayClassModel(AbstractModel):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1", class_weights=None, **config):
        """
        An essay is represented by the first 512 tokens
        class_weights: Dict[name]=torch.Tesnor([weights])
        """
        super().__init__(class_nums, class_weights, **config)
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        if self.config["label_smoothing"]:
            self.losses = {name: LabelSmoothingLoss(len(lst), smoothing=self.config["smoothing"], weight=self.class_weights[name]) for name, lst in class_nums.items()}

    def forward(self, batch):
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True).cuda()
        enc = self.bert(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        return {name: layer(enc.pooler_output) for name, layer in self.cls_layers.items()}


class ProjectionClassModel(AbstractModel):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1", class_weights=None, **config):
        super().__init__(class_nums, class_weights, **config)
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.proj_layers=torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size) for name, lst in class_nums.items()})
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        if self.config["label_smoothing"]:
            self.losses = {name: LabelSmoothingLoss(len(lst), smoothing=self.config["smoothing"], weight=self.class_weights[name]) for name, lst in class_nums.items()}

    def forward(self, batch):
        enc = self.bert(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        masked_enc=enc.last_hidden_state*batch['attention_mask'].unsqueeze(-1)
        avg_end=torch.sum(masked_enc,1)/torch.sum(batch["attention_mask"],-1).unsqueeze(-1)
        result={}
        for name, layer in self.cls_layers.items():
            projected=torch.tanh(self.proj_layers[name](avg_end))
            result[name]=layer(projected)
        return result
