import argparse
import pytorch_lightning as pl
import sys
import eb_data_reader, eb_model


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_checkpoint', default=None)
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--tsvs',nargs="+",help="Tsv(s) with the data")


    args=parser.parse_args()

    data=eb_data_reader.RowDataModule(args.tsvs,batch_size=args.batch_size,bert_model_name=args.bert_path)
    data.setup()

    model=eb_model.ClassModel(data.class_nums(),bert_model=args.bert_path)
    trainer=pl.Trainer(gpus=1,max_epochs=args.epochs)
    trainer.fit(model,datamodule=data)
    
