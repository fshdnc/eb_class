import argparse
import datetime
import pytorch_lightning as pl
import sys
from finnessayscore import data_reader
from finnessayscore import model
import os
from pytorch_lightning.callbacks import ModelCheckpoint
#from transformers import TrainingArguments

# for debugging
def print_n_return(item):
    print(item)
    return item

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_label_smoothing', default=False, action="store_true", help="Use label smoothing")
    parser.add_argument('--smoothing', type=float, default=0, help="0: one-hot method, 0<x<1: smooth method")
    parser.add_argument('--jsons',nargs="+",help="JSON(s) with the data")
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--whole_essay_overlap', type=int, default=10)
    parser.add_argument('--model_type', default="sentences", help="trunc_essay, whole_essay, seg_essay, or sentences")
    parser.add_argument('--max_length', type=int, default=512, help="max number of token used in the whole essay model")
    parser.add_argument('--run_id', help="Optional run id")

    args = parser.parse_args()
    assert args.model_type in ["whole_essay", "sentences", "trunc_essay", "seg_essay"]
    if args.model_type=="sentences":
        for j in args.jsons:
            assert "parsed" in j
    if args.use_label_smoothing:
        assert args.smoothing <1 and args.smoothing>=0
    print(args)
    if not args.run_id:
        args.run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")
    print("RUN_ID", args.run_id, sep="\t")


    data = data_reader.JsonDataModule(args.jsons,
                                      batch_size=args.batch_size,
                                      bert_model_name=args.bert_path,
                                      model_type=args.model_type,
                                      stride=args.whole_essay_overlap,
                                      max_token=args.max_length)
    data.setup()
    train_len, dev_len, test_len = data.data_sizes()

    class_weights = data.get_class_weights()

    if args.model_type=="whole_essay":
        m = model.WholeEssayClassModel
    elif args.model_type=="sentences":
        m = model.ClassModel
    elif args.model_type in ["trunc_essay", "seg_essay"]:
        m = model.TruncEssayClassModel
    #model = model.ProjectionClassModel(data.class_nums(),
    model = m(data.class_nums(),
              bert_model=args.bert_path,
              lr=args.lr,
              label_smoothing=args.use_label_smoothing, smoothing=args.smoothing,
              num_training_steps=train_len//args.batch_size*args.epochs,
              class_weights={k: v.cuda() for k, v in class_weights.items()})
    #os.system("rm -rf lightning_logs")
    logger = pl.loggers.TensorBoardLogger("lightning_logs",
                                          name=args.run_id,
                                          version="latest")
    checkpoint_callback = ModelCheckpoint(monitor='val_acc_lab_grade',
                                          filename="baseline-{epoch:02d}-{val_acc_lab_grade:.2f}",
                                          save_top_k=1,
                                          mode="max")
    trainer = pl.Trainer(gpus=1,
                         accumulate_grad_batches=args.grad_acc,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=0,
                         log_every_n_steps=1,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         fast_dev_run=False)
    trainer.fit(model, datamodule=data)
    checkpoint_callback.best_model_path

    model.eval()
    model.cuda()

    from finnessayscore.evaluate import evaluate
    print("Training set")
    evaluate(data.train_dataloader(), model, data.get_label_map(), model_type=args.model_type)
    print("Validation set")
    evaluate(data.val_dataloader(), model, data.get_label_map(), model_type=args.model_type, plot_conf_mat=True)
    

