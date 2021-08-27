import os, sys, argparse
import datetime
import torch
import numpy
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from finnessayscore import data_reader
from finnessayscore.model import AbstractModel
from finnessayscore.evaluate import plot_confusion_matrix

class TFIDFModel(AbstractModel):

    def __init__(self, class_nums, data, class_weights=None, **config):
        """
        class_weights: Dict[name]=torch.Tesnor([weights])
        """
        super().__init__(class_nums, class_weights, **config)
        self.vectorizer = TfidfVectorizer(ngram_range=(2,5), analyzer="char_wb") #,stop_words=stop_words)
        self.vectorizer.fit(data)
        self.cls_layers = torch.nn.ModuleDict({name: torch.nn.Linear(len(self.vectorizer.idf_), len(lst)) for name, lst in class_nums.items()})

    def forward(self, batch):
        enc = self.vectorizer.transform(batch["essay"])
        enc = torch.from_numpy(enc.toarray()).float().to(batch["essay"].device)
        return {name: layer(enc) for name, layer in self.cls_layers.items()}
        
def evaluate(dataloader, model, label_map, plot_conf_mat=False, fname=None):
    # only the model input differs from the regular eval
    with torch.no_grad():
        preds = []
        target = []
        for batch in dataloader:
            output = model({k:v for k, v in batch.items() if k=="essay"})
            preds.append(output) #["lab_grade"])
            target.append(batch["lab_grade"])

    preds = [v for item in preds for k,vs in item.items() for v in vs]
    preds = [int(torch.argmax(pred)) for pred in preds]
    preds = [label_map["lab_grade"][p] for p in preds]
    target = [int(tt) for t in target for tt in t]
    target = [label_map["lab_grade"][p] for p in target]
    print("Predictions:", preds)
    print("Gold standard:", target)
    assert len(preds)==len(target)
    corrects = [1 if p==t else 0 for p, t in zip(preds,target)]
    print("Acc\t{}".format(sum(corrects)/len(corrects)))
    print("Predicted class number:",len(set(preds)))
    conf_mat = confusion_matrix(preds, target, labels=[l for i, l in label_map["lab_grade"].items()])
    print("Confusion matrix:\n", conf_mat)
    if plot_conf_mat:
        plot_confusion_matrix(conf_mat, label_map["lab_grade"], fname=fname)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--jsons',nargs="+",help="JSON(s) with the data")


    args = parser.parse_args()
    run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")

    data = data_reader.JsonDataModule(args.jsons,
                                      batch_size=args.batch_size,
                                      bert_model_name=args.bert_path)
    data.setup()
    train_len, dev_len, test_len = data.data_sizes()

    class_weights = data.get_class_weights()

    model = TFIDFModel(data.class_nums(),
                                      [d["essay"] for d in data.train],
                                      lr=args.lr,
                                      num_training_steps=train_len//args.batch_size*args.epochs,
                                      class_weights={k: v for k, v in class_weights.items()})
    os.system("rm -rf lightnint_logs")
    logger = pl.loggers.TensorBoardLogger("lightning_logs",
                                          name="baseline",
                                          version=run_id)
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_lab_grade',
                                          filename="{epoch:02d}-{val_acc_lab_grade:.2f}",
                                          save_top_k=1,
                                          mode="max")
    trainer = pl.Trainer(gpus=1,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=1,
                         log_every_n_steps=1,
                         logger=logger,
                         callbacks=[checkpoint_callback])
    trainer.fit(model, datamodule=data)

    model.eval()

    print("Training set")
    evaluate(data.train_dataloader(), model, data.get_label_map())
    print("Validation set")
    evaluate(data.val_dataloader(), model, data.get_label_map(), plot_conf_mat=True, fname=run_id)


    

