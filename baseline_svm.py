import os, sys, argparse
import datetime
import numpy
import sklearn.svm
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import data_reader
from evaluate import plot_confusion_matrix

def evaluate(dataloader, model, label_map, plot_conf_mat=False, fname=None):
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


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert_path', default='TurkuNLP/bert-base-finnish-cased-v1')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--jsons',nargs="+",help="JSON(s) with the data")


    args = parser.parse_args()
    run_id = str(datetime.datetime.now()).replace(":","").replace(" ","_")

    data = data_reader.JsonDataModule(args.jsons,
                                      batch_size=args.batch_size,
                                      bert_model_name=args.bert_path)
    data.setup()
    train_len, dev_len, test_len = data.data_sizes()

    # vectorize the data
    X_train = [d["essay"][:1500] for d in data.train]
    Y_train = [d["lab_grade"] for d in data.train]
    X_dev = [d["essay"][:1500] for d in data.dev]
    Y_dev = [d["lab_grade"] for d in data.dev]
    vectorizer = TfidfVectorizer(ngram_range=(2,5), analyzer="char_wb") #,stop_words=stop_words)
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_dev = vectorizer.transform(X_dev)

    # train an SVM
    class_weights = data.get_class_weights()["lab_grade"].numpy()
    class_weights = {i: w for i, w in enumerate(class_weights)}
    sample_weights = numpy.array([class_weights[gold] for gold in Y_train])
    svm_classifier = sklearn.svm.LinearSVC(max_iter=10000)
    svm_classifier.fit(X_train, Y_train, sample_weight=sample_weights)

    label_map =data.get_label_map()
    preds = svm_classifier.predict(X_dev)
    preds = [label_map["lab_grade"][p] for p in preds]
    target = Y_dev
    target = [label_map["lab_grade"][p] for p in target]
    assert len(target)==len(preds)
    corrects = [1 if p==t else 0 for p, t in zip(preds, target)]
    print("Acc\t{}".format(sum(corrects)/len(corrects)))
    print("Predicted class number:",len(set(preds)))
    conf_mat = confusion_matrix(preds, target, labels=[l for i, l in label_map["lab_grade"].items()])
    print("Confusion matrix:\n", conf_mat)
    plot_confusion_matrix(conf_mat, label_map["lab_grade"], fname="baseline-tfidf-svm.png")


    

