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
        self.class_nums = class_nums
        self.train_acc = torch.nn.ModuleDict({name: torchmetrics.Accuracy() for name in class_nums})
        self.val_acc = torch.nn.ModuleDict({name: torchmetrics.Accuracy() for name in class_nums})
        self.train_qwk = torch.nn.ModuleDict({name: torchmetrics.CohenKappa(num_classes=len(lst), weights='quadratic') for name, lst in class_nums.items()})
        self.val_qwk = torch.nn.ModuleDict({name: torchmetrics.CohenKappa(num_classes=len(lst), weights='quadratic') for name, lst in class_nums.items()})
        # Unfortunately there is no BufferDict yet so we just simulate with name mangling
        if class_weights is not None:
            for name, weights in class_weights.items():
                self.register_buffer("class_weights__" + name, weights)

        class _ClassWeights:
            def __getitem__(_, key):
                getattr(self, "class_weights__" + key, None)
        self.class_weights = _ClassWeights()
        self.config = config

    def out_to_cls(self, out):
        return {name: torch.argmax(pred, 1) for name, pred in out.items()}

    def compute_loss(self, name, pred, gold, weight):
        if self.config["label_smoothing"]:
            return self.losses[name](pred, gold)
        else:
            return F.cross_entropy(pred, gold, weight=weight)

    def training_step(self,batch,batch_idx):
        out = self(batch)
        out_cls = self.out_to_cls(out)
        losses = []
        pbar = {}
        for name in self.class_nums:
            loss = self.compute_loss(name, out[name], batch[name], self.class_weights[name])
            losses.append(loss)
            acc = self.train_acc[name](out_cls[name], batch[name])
            self.log(f'train_acc_{name}', acc*100)
            self.log(f'train_loss_{name}', loss)
            pbar["acc_"+name] = f"{acc*100:03.1f}"
            qwk = self.train_qwk[name](out_cls[name], batch[name])
            self.log(f'train_qwk_{name}', qwk)
        return {"loss":sum(losses), "progress_bar":pbar}

    def training_epoch_end(self, _):
        for name in self.class_nums:
            print(f"train_acc_{name}", self.train_acc[name].compute()*100)
            self.log(f"train_acc_{name}", self.train_acc[name].compute()*100)
            self.train_acc[name].reset()
            print(f"train_qwk_{name}", self.train_qwk[name].compute()*100)
            self.log(f"train_qwk_{name}", self.train_qwk[name].compute()*100)
            self.train_qwk[name].reset()

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        out_cls = self.out_to_cls(out)
        #losses = []
        for name in self.class_nums:
            loss = self.compute_loss(name, out[name], batch[name], self.class_weights[name])
            #losses.append(loss)
            self.log(f'train_loss_{name}', loss)
            self.val_acc[name](out_cls[name], batch[name])
            self.val_qwk[name](out_cls[name], batch[name])

    def validation_epoch_end(self, _):
        for name in self.class_nums:
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
        #self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        self.final_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, len(lst)) for name, lst in class_nums.items()})
        if self.config["label_smoothing"]:
            self.losses = {name: LabelSmoothingLoss(len(lst), smoothing=self.config["smoothing"], weight=self.class_weights[name]) for name, lst in class_nums.items()}

    def forward(self, batch):
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            batch[k] = torch.nn.utils.rnn.pad_sequence(batch[k], batch_first=True)
        enc = self.bert(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        if "pooling" in self.config:
            if self.config["pooling"]=="mean":
                return {name: layer(torch.mean(enc.last_hidden_state, axis=1)) for name, layer in self.final_layers.items()}
            elif self.config["pooling"]=="cls":
                return {name: layer(enc.pooler_output) for name, layer in self.final_layers.items()}
            else:
                raise ValueError("Pooling method specified not known.")
        else: #defaults to cls
            return {name: layer(enc.pooler_output) for name, layer in self.final_layers.items()}


class TruncEssayOrdModel(AbstractModel):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1", class_weights=None, **config):
        if config["label_smoothing"]:
            raise NotImplementedError("label_smoothing not implemented for TruncEssayOrdModel")
        super().__init__(class_nums, class_weights, **config)
        self.bert = transformers.BertModel.from_pretrained(bert_model)
        self.reg_layers = torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, 1, bias=False) for name in class_nums.keys()})
        self.cutoffs = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(len(lst) - 1).float())
            for name, lst in class_nums.items()
        })
        # Init cutoffs as ordered
        for name in self.cutoffs:
            torch.nn.init.normal_(self.cutoffs[name])
            sort_param_inplace(self.cutoffs[name])

    def out_to_cls(self, out):
        return {name: (pred >= 0).sum(axis=1) for name, pred in out.items()}

    def score_to_cls(self, scores):
        return self.out_to_cls({
            name: cont_score - self.cutoffs[name]
            for name, cont_score in scores.items()
        })

    def compute_loss(self, name, pred, gold, weight):
        num_classes = len(self.class_nums[name])
        gold_cut = torch.vstack(
            [gold >= cut for cut in range(1, num_classes)]
        ).T.float()
        """
        TODO: Check if this makes sense and add it back in
        print("weight", weight)
        weight_gte_sum = torch.flip(
            torch.cumsum(
                torch.flip(weight, (0,))[:-1], 0
            ), (0,)
        )
        weight_lt_sum = torch.cumsum(weight[:-1], 0)
        pos_mass = weight_gte_sum / torch.arange(num_classes - 1, 0, -1)
        neg_mass = weight_lt_sum / torch.arange(1, num_classes)
        pos_weight = pos_mass / neg_mass
        pos_weight[(pos_weight == 0) | torch.isinf(pos_weight)] = 1
        print("pos_weight", pos_weight)
        return F.binary_cross_entropy_with_logits(pred, gold_cut, pos_weight=pos_weight)
        """
        return F.binary_cross_entropy_with_logits(pred, gold_cut)

    def _forward_enc(self, batch):
        for k in ["input_ids", "attention_mask", "token_type_ids"]:
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                batch[k],
                batch_first=True
            )
        enc = self.bert(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        token_type_ids=batch['token_type_ids'])
        return enc

    def _score_out(self, enc):
        return {
            name: layer(enc.pooler_output)
            for name, layer in self.reg_layers.items()
        }

    def forward_score(self, batch):
        return self._score_out(self._forward_enc(batch))

    def cutoffs_score_scale(self):
        return self.cutoffs

    def forward(self, batch):
        return {
            name: cont_score - self.cutoffs[name]
            for name, cont_score in self.forward_score(batch).items()
        }


def sort_param_inplace(param):
    param.data.copy_(torch.sort(param)[0])


class PedanticTruncEssayOrdModel(TruncEssayOrdModel):

    def __init__(
        self,
        class_nums,
        bert_model="TurkuNLP/bert-base-finnish-cased-v1",
        class_weights=None,
        **config
    ):
        super().__init__(class_nums, bert_model, class_weights, **config)
        # Norm then allow changing scale but keep mean centered
        self.norm = torch.nn.ModuleDict({
            name: torch.nn.BatchNorm1d(1, affine=False)
            for name in class_nums.keys()
        })
        self.discrim = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.ones(1).float())
            for name in class_nums.keys()
        })

    def _score_out(self, enc):
        return {
            name: self.norm[name](layer(enc.pooler_output))
            for name, layer in self.reg_layers.items()
        }

    def cutoffs_score_scale(self):
        return {
            name: cutoff / self.discrim[name]
            for name, cutoff in self.cutoffs.items()
        }

    def forward(self, batch):
        return {
            name: self.discrim[name] * cont_score_norm - self.cutoffs[name]
            for name, cont_score_norm in self.forward_score(batch).items()
        }

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # Keep cutoffs ordered => output stays positively correlated with grade
        for name in self.cutoffs:
            sort_param_inplace(self.cutoffs[name])


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
