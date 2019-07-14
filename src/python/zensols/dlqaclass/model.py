import logging
import sys
from time import time
import numpy as np
import itertools as it
import csv
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from zensols.actioncli import persisted
from zensols.dltools import CudaConfig
from zensols.dlqaclass import (
    QADataLoader,
    Net,
)

logger = logging.getLogger(__name__)


class IrDataset(object):
    def __init__(self, data, cuda):
        self.data = data
        self.cuda = cuda

    def __getitem__(self, i):
        # paragraph, question, label, paragraph ID
        p = self.data[i]
        lab = self.cuda.singleton(p[2], dtype=torch.long)
        i = self.cuda.singleton(p[0].id, dtype=torch.long)
        mats = (p[0].matrix, p[1].matrix,
                p[0].feature_matrix, p[1].feature_matrix,
                p[0].common_matrix(p[1]))
        return (mats, lab, i,)

    def __len__(self):
        return len(self.data)


class QAModelManager(object):
    SECTION = 'nn_model'

    def __init__(self, config):
        self.config = config
        # binary classification
        self.n_labels = 2
        # CUDA configuration resource
        self.cuda = CudaConfig()
        # model parameters
        self.cnf = config.populate(section=self.SECTION)
        # whether or not to debug the network
        self.debug = self.cnf.debug
        # location of where to store and load the model
        self.model_path = config.get_option_path('model_path', self.SECTION)
        # results paths
        self.validation_path = config.get_option_path('validation_path', self.SECTION)
        self.test_path = config.get_option_path('test_path', self.SECTION)
        self.pred_path = config.get_option_path('pred_path', self.SECTION)
        if self.debug:
            logger.setLevel(logging.DEBUG)

    @property
    @persisted('_data_loader')
    def data_loader(self):
        return QADataLoader(self.config)

    @property
    def dataset(self):
        loader = self.data_loader
        return tuple(map(lambda x: loader.get_dataset(x)[0],
                         'train test'.split()))

    @property
    def ir_datasets(self):
        cuda = CudaConfig(use_cuda=False)
        logger.debug(f'creating dataset...')
        train, test = self.dataset
        train, test = IrDataset(train, cuda), IrDataset(test, cuda)
        logger.debug(f'created datasets')
        return train, test

    @property
    def dataloaders(self):
        train, test = self.ir_datasets
        self.train_dataset = train
        self.test_dataset = test
        # obtain training indices that will be used for validation
        num_train = len(train)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.cnf.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]
        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        logger.debug(f'preparing data loaders')
        # prepare data loaders
        train_loader = DataLoader(
            train, batch_size=self.cnf.batch_size, sampler=train_sampler)
        valid_loader = DataLoader(
            train, batch_size=self.cnf.batch_size, sampler=valid_sampler)
        test_loader = DataLoader(
            test, batch_size=self.cnf.batch_size)
        logger.debug(f'created loaders')
        return train_loader, valid_loader, test_loader

    def create_model(self):
        ds = self.dataset
        ds_idx = 1 if len(ds[0]) == 0 else 0
        point = ds[ds_idx][0]
        logger.debug(f'ds: {type(point[0])}')
        logger.debug(f'ds: {type(point[1])}')
        para_shape = point[0].matrix.shape
        ques_shape = point[1].matrix.shape
        para_f_shape = point[0].feature_matrix.shape
        ques_f_shape = point[1].feature_matrix.shape
        logger.debug(f'shapes: paragraph {para_shape}, question: {ques_shape}')
        model = Net(para_shape, ques_shape, para_f_shape, ques_f_shape,
                    self.n_labels, self.cnf, self.debug)
        return self.cuda.to(model)

    def create_optimizer_criterion(self, model):
        # opt = torch.optim.SGD(model.parameters(), lr=self.cnf.learning_rate)
        opt = optim.Adam(model.parameters(), lr=self.cnf.learning_rate)
        loss = nn.CrossEntropyLoss()
        #loss = nn.NLLLoss()
        return opt, loss

    def _write_validation(self, train_loss, validation_loss,
                          decreased, mode='a'):
        if self.validation_path is not None:
            self.validation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.validation_path, mode) as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow((train_loss, validation_loss, decreased))

    def train(self):
        logger.info('training...')
        bail_on_early_stop = True
        logger.debug(f'loading corpus')
        train, valid, test = self.dataloaders
        logger.debug(f'created all three dataloaders')
        if self.debug:
            epochs = 1
            max_training = 1
            do_validate = False
        else:
            epochs = self.cnf.epochs
            max_training = sys.maxsize
            do_validate = True
        logger.debug('creating model')
        model = self.create_model()
        if model is None:
            return
        logger.debug('creating optimizer and criterion')
        optimizer, criterion = self.create_optimizer_criterion(model)
        # set initial "min" to infinity
        valid_loss_min = np.Inf
        self._write_validation(*'train validation decreased'.split(), mode='w')
        t0 = time()
        logger.debug(f'training with {epochs} epochs')
        for epoch in range(epochs):
            logger.debug(f'starting epoc: {epoch}')
            # keep track of training and validation loss
            train_loss = 0.0
            valid_loss = 0.0
            dl_data = it.islice(enumerate(train), max_training)
            for i, (mats, labels, pid) in dl_data:
                logger.debug(f'data: {len(mats)} {mats[0].shape}')
                mats = tuple(map(self.cuda.to, mats))
                optimizer.zero_grad()
                # forward pass, get our log probs
                try:
                    output = model(mats)
                    # calculate the loss with the logps and the labels
                    loss = criterion(output, labels)
                except Exception as e:
                    print(e)
                    return
                loss.backward()
                # update/iterate over the error surface
                optimizer.step()
                train_loss += loss.item() * labels.size(0)
                logger.debug(f'loss ({i}): {train_loss}')
                if i > 0 and ((i % 30) == 0):
                    logger.info(f'{i}; train loss={train_loss / i}')
                    #gc.collect()
            if do_validate:
                # prep model for evaluation
                model.eval()
                for mats, labels, pid in valid:
                    logger.debug(f'data: {len(mats)} {mats[0].shape}')
                    mats = tuple(map(self.cuda.to, mats))
                    # forward pass: compute predicted outputs by passing
                    # inputs to the model
                    output = model(mats)
                    # calculate the loss
                    loss = criterion(output, labels)
                    # update running validation loss
                    valid_loss += loss.item() * labels.size(0)
                # calculate average loss over an epoch
                train_loss = train_loss / len(train)
                valid_loss = valid_loss / len(valid)
                decrease = valid_loss <= valid_loss_min
                logger.info(f'epoch: {epoch+1}, training loss: {train_loss:.6f}, ' +
                            f'validation Loss: {valid_loss:.6f}')
                self._write_validation(train_loss, valid_loss, str(decrease).lower())
                # save model if validation loss has decreased
                if decrease:
                    logger.info(f'validation loss decreased ' +
                                f'({valid_loss_min:.6f} --> {valid_loss:.6f})')
                    logger.info(f'saving model to {self.model_path}')
                    model_file = str(self.model_path.absolute())
                    self.model_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), model_file)
                    valid_loss_min = valid_loss
                elif bail_on_early_stop:
                    break
        logger.info(f'trained in {(time() - t0):.3f}s')
        self.cuda.empty_cache()
        return model

    def test(self, model, train_start=None, writer=sys.stdout):
        logger.info('testing...')
        test_start = time()
        if train_start is None:
            train_start = test_start
        train, valid, test = self.dataloaders
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        class_correct = list(0. for i in range(self.n_labels))
        class_total = list(0. for i in range(self.n_labels))
        optimizer, criterion = self.create_optimizer_criterion(model)
        # prep model for evaluation
        model.eval()
        for mats, labels, pid in test:
            logger.debug(f'data: {len(mats)} {mats[0].shape}')
            mats = tuple(map(self.cuda.to, mats))
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                output = model(mats)
            # calculate the loss
            loss = criterion(output, labels)
            # update test loss
            test_loss += loss.item() * labels.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            logger.debug(f'labels: {labels}')
            logger.debug(f'correct: {correct}')
            # calculate test accuracy for each object class
            for i in range(len(labels)):
                label = labels.data[i]
                logger.debug(f'label: {label}, {class_correct[label]}')
                class_correct[label] += correct[i].item()
                class_total[label] += 1
        # calculate and print avg test loss
        test_loss = test_loss / len(test.dataset)
        logger.info(f'test Loss: {test_loss:.6f}')
        for i in range(self.n_labels):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                writer.write(f'test accuracy of label {i}: {acc:2.0f}% ' +
                             f'({np.sum(class_correct[i]):2.0f}/' +
                             f'{np.sum(class_total[i]):2.0f})\n')
            else:
                writer.write(f'test accuracy of {i}: no training examples\n')
        acc = 100. * np.sum(class_correct) / np.sum(class_total)
        writer.write(f'test accuracy (overall): {acc:2.0f}% ' +
                     f'({np.sum(class_correct):2.0f}/' +
                     f'{np.sum(class_total):2.0f})\n')
        time_train = test_start - train_start
        time_test = time() - test_start
        time_all = time() - train_start
        writer.write(f'time: train: {time_train:.1f}s, ' +
                     f'test: {time_test:.1f}s, ' +
                     f'all: {time_all:.1f}s\n')
        self.cuda.empty_cache()

    def load_model(self):
        model_file = str(self.model_path.absolute())
        logger.info(f'loading model from {model_file}')
        state = torch.load(model_file)
        model = self.create_model()
        model.load_state_dict(state)
        return model

    def predict(self, model):
        train, valid, test = self.dataloaders
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        optimizer, criterion = self.create_optimizer_criterion(model)
        # prep model for evaluation
        model.eval()
        preds = []
        for data, labels, pids in test:
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                output = model(data)
            # calculate the loss
            loss = criterion(output, labels)
            # update test loss
            test_loss += loss.item() * data.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            proba = torch.exp(output[:, 1])
            pred_data = torch.stack((proba, labels.float(), correct.float(), pids.float()))
            pred_data = pred_data.transpose(0, 1)
            preds.append(pred_data)
        preds = torch.cat(preds)
        preds = preds.cpu().clone().detach()
        td = self.ir_datasets[1].data
        # by dataset ID, get the paragraph and section (article) IDs
        pdat = map(lambda i: (td[i][0].id, td[i][0].tid, td[i][0].tid == td[i][1].paragraph.tid),
                   preds[:, 3].int())
        pdat = torch.tensor(tuple(pdat), dtype=self.cuda.data_type)
        preds = torch.cat((preds[:, :3], pdat), 1)
        para_acc = preds[:, 2].sum() / preds.shape[0]
        # filter by matching article
        match_articles = preds[preds[:, 5].nonzero().squeeze()]
        # filter on positive labels (matching sections should predict true)
        article_acc = match_articles[:, 1].sum() / match_articles.shape[0]
        print(f'paragraph accuracy: {para_acc}, article accuracy: {article_acc}')
        self.pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pred_path, 'w') as f:
            f.write(f'Probability,Label,Correct,Paragraph ID,Article ID,Matching Article\n')
            np.savetxt(f, preds.numpy(), fmt='%2.2f', delimiter=',')

    def rank_question(self, model, ques, paras, limit=sys.maxsize):
        #eps = 1e-6
        eps = None
        logger.info(f'ranking {ques}')
        gold_para = ques.paragraph
        ds = map(lambda p: (p, ques, False),
                 filter(lambda p: p.id != gold_para.id, paras))
        ds = list(it.islice(ds, limit))
        ds.append((gold_para, ques, True))
        dl = DataLoader(IrDataset(ds, CudaConfig(use_cuda=False)),
                        batch_size=self.cnf.batch_size)
        optimizer, criterion = self.create_optimizer_criterion(model)
        # prep model for evaluation
        model.eval()
        preds = []
        # initialize lists to monitor test loss and accuracy
        test_loss = 0.0
        t0 = time()
        for mats, labels, pids in dl:
            logger.debug(f'data: {len(mats)} {mats[0].shape}')
            mats = tuple(map(self.cuda.to, mats))
            # forward pass: compute predicted outputs by passing inputs
            # to the model
            with torch.no_grad():
                output = model(mats)
            # calculate the loss
            loss = criterion(output, labels)
            # update test loss
            test_loss += loss.item() * labels.size(0)
            # convert output probabilities to predicted class
            _, pred = torch.max(output, 1)
            # compare predictions to true label
            correct = np.squeeze(pred.eq(labels.data.view_as(pred)))
            probs = torch.exp(output)
            proba = probs[:, 1]
            if eps is not None:
                ps = probs.sum(dim=1)
                tol = ps[torch.nonzero(abs(1.0 - ps) > eps)]
                if tol.shape[0] > 0:
                    logger.warning(f'N predictions of binary probabilities ' +
                                   f'not in error ({eps}): {tol.shape[0]}')
            pred_data = torch.stack(
                (proba, labels.float(), correct.float(), pids.float()))
            pred_data = pred_data.transpose(0, 1)
            preds.append(pred_data.cpu())
        preds = torch.cat(preds)
        _, indicies = torch.sort(preds, 0, descending=True)
        preds = preds[indicies[:, 0]]
        logger.debug(f'gold paragraph: {gold_para.id}')
        rank_row = torch.nonzero(preds[:, 3] == gold_para.id)
        rank_idx = int(rank_row[0][0])
        logger.info(f'calc rank: {rank_idx} in {time()-t0:.2f}s')
        return {'rank': rank_idx,
                'paragraph_id': gold_para.id,
                'question_id': ques.id,
                'n_paragraphs': len(ds),
                'preds': preds}

    def tmp(self):
        if 0:
            model = self.train()
        else:
            model = self.load_model()
            self.test(model)
