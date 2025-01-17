import pickle
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
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--use_label_smoothing', default=False, action="store_true", help="Use label smoothing")
    parser.add_argument('--smoothing', type=float, default=0, help="0: one-hot method, 0<x<1: smooth method")
    parser.add_argument('--grad_acc', type=int, default=1)
    parser.add_argument('--pooling', default="cls", help="only implemented for trunc_essay model, cls or mean")
    parser.add_argument('--run_id', help="Optional run id")
    #parser.add_argument('--gpus', default=None, help="Number of gpus")

    pl.Trainer.add_argparse_args(parser)
    data_reader.JsonDataModule.add_argparse_args(parser)

    args = parser.parse_args()
    if args.use_label_smoothing:
        assert args.smoothing <1 and args.smoothing>=0
    print(args)
    if not args.run_id:
        args.run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")
    print("RUN_ID", args.run_id, sep="\t")
    os.mkdir(args.run_id)

    data = data_reader.JsonDataModule.from_argparse_args(args)
    data.setup()
    train_len, dev_len, test_len = data.data_sizes()

    class_weights = data.get_class_weights()

    if args.model_type=="whole_essay":
        m = model.WholeEssayClassModel
    elif args.model_type=="sentences":
        m = model.ClassModel
    elif args.model_type == "trunc_essay_ord":
        m = model.TruncEssayOrdModel
    elif args.model_type == "pedantic_trunc_essay_ord":
        m = model.PedanticTruncEssayOrdModel
    elif args.model_type in ["trunc_essay", "seg_essay"]:
        m = model.TruncEssayClassModel
    #model = model.ProjectionClassModel(data.class_nums(),
    model = m(data.class_nums(),
              bert_model=args.bert_model,
              lr=args.lr,
              label_smoothing=args.use_label_smoothing, smoothing=args.smoothing,
              num_training_steps=train_len//args.batch_size*args.epochs,
              class_weights=class_weights,
              pooling=args.pooling)
    #os.system("rm -rf lightning_logs")
    logger = pl.loggers.TensorBoardLogger("lightning_logs",
                                          name=args.run_id,
                                          version="latest")
    checkpoint_callback = ModelCheckpoint(monitor='val_qwk_lab_grade',
                                          filename="baseline-{epoch:02d}-{val_acc_qwk_grade:.2f}",
                                          save_top_k=1,
                                          mode="max")
    trainer = pl.Trainer.from_argparse_args(
        args,
        accumulate_grad_batches=args.grad_acc,
        max_epochs=args.epochs,
        progress_bar_refresh_rate=0,
        log_every_n_steps=1,
        logger=logger,
        callbacks=[checkpoint_callback],
        #gpus=args.gpus
    )
    trainer.fit(model, datamodule=data)
    checkpoint_callback.best_model_path

    model.eval()
    if trainer.gpus is not None and trainer.gpus > 0:
        model.cuda()

    from finnessayscore.evaluate import evaluate
    print("Training set")
    evaluate(data.train_dataloader(), model, data.get_label_map(), model_type=args.model_type)
    print("Validation set")
    evaluate(
        data.val_dataloader(),
        model,
        data.get_label_map(),
        model_type=args.model_type,
        plot_conf_mat=True,
        do_plot_beeswarm=args.model_type.endswith("_ord"),
        do_plot_prob=True,
        fname=args.run_id+"/"
    )
    

