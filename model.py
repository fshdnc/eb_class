import pytorch_lightning as pl
import transformers

class ClassModel(pl.LightningModule):

    def __init__(self, class_nums, bert_model="TurkuNLP/bert-base-finnish-cased-v1"):
        super().__init__()
        self.bert=transformers.BertModel.from_pretrained(bert_model)
        self.cls_layers=torch.nn.ModuleDict({name: torch.nn.Linear(self.bert.config.hidden_size, n) for name, n in class_nums.items()})
        self.accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()


    def forward(self,batch):
        enc=self.bert(input_ids=batch['input_ids'],
                      attention_mask=batch['attention_mask'],
                      token_type_ids=batch['token_type_ids']) #BxS_LENxSIZE; BxSIZE
        return {name: layer(enc.pooler_output) for name, layer in self.cls_layers.items()}

    def training_step(self,batch,batch_idx):
        out = self(batch)
        losses=[F.cross_entropy(out[name],batch[name]) for name in self.cls_layers]
        return losses

    def validation_step(self,batch,batch_idx):
        out = self(batch)
        losses=[F.cross_entropy(out[name],batch[name]) for name in self.cls_layers]
        self.log(losses)

    def configure_optimizers(self):
        optimizer = transformers.optimization.AdamW(self.parameters(), lr=1e-5)
        scheduler = transformers.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(self.steps_train*0.1), num_training_steps=self.steps_train)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]
