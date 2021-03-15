import argparse
import pytorch_lightning as pl
import sys
import data_reader, model
import os

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
    parser.add_argument('--jsons',nargs="+",help="JSON(s) with the data")


    args = parser.parse_args()

    data = data_reader.JsonDataModule(args.jsons,
                                      batch_size=args.batch_size,
                                      bert_model_name=args.bert_path)
    data.setup()
    train_len, dev_len, test_len = data.data_sizes()

    model = model.ClassModel(data.class_nums(),
                             bert_model=args.bert_path,
                             lr=1e-6,
                             num_training_steps=train_len//args.batch_size*args.epochs)
    os.system("rm -rf lightnint_logs")
    logger = pl.loggers.TensorBoardLogger("lightning_logs", version="latest", name="ebm")
    trainer = pl.Trainer(gpus=1,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=1,
                         log_every_n_steps=1,
                         logger=logger)
    trainer.fit(model, datamodule=data)

    model.eval()
    model.cuda()

    #evaluate(dataloader, dataset, model, model_output_to_p, save_directory='plots')
    #evaluate(dataloader, dataset, model, model_output_to_p, save_directory=None):
    # TODO: instead of tensor
    from evaluate import evaluate
    print("Validation set")
    evaluate(data.val_dataloader(), model)
    print("Training set")
    evaluate(data.train_dataloader(), model)


    

