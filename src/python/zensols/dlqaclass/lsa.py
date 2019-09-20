import logging
import itertools as it
import sys
from multiprocessing import Pool
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from zensols.actioncli import (
    time,
    persisted,
    PersistedWork,
)
from zensols.nlp import TokenNormalizerFactory
from zensols.dlqaclass import (
    AppConfig,
    QADataLoader,
)

logger = logging.getLogger(__name__)


def feat_to_tokens(feat):
    return tuple(map(lambda x: x[0]['lemma'].lower(),
                     filter(lambda x: not x[1]['is_stop'],
                            zip(feat.str_feats, feat.feats))))


class DocModelManager(object):
    SECTION = 'lsa_model'

    def __init__(self, config):
        self.config = config
        # model parameters
        self.cnf = config.populate(section=self.SECTION)
        path = self.config.get_option_path('model_path', self.SECTION)
        self._model = PersistedWork(path, self, cache_global=True)
        self._rank_multi = PersistedWork('_rank_multi', self, cache_global=True)
        res_path = self.config.get_option_path('result_path', self.SECTION)
        self._ranks = PersistedWork(res_path, self)

    @property
    @persisted('_dataloader', cache_global=True)
    def dataloader(self):
        return QADataLoader(self.config)

    @property
    def paragraph_train_features(self):
        return self.dataloader.get_feature_sets('train')

    @property
    def questions(self):
        pfs = self.paragraph_train_features
        return it.chain(*map(lambda p: p.questions, pfs))

    @property
    @persisted('_questions_by_id', cache_global=True)
    def questions_by_id(self):
        return {p.id: p for p in self.questions}

    @property
    @persisted('_model')
    def model(self):
        pfs = self.dataloader.get_feature_sets('train')
        logger.info(f'vectorizing {len(pfs)} paragraphs')
        Y_train = tuple(map(lambda p: p.id, pfs))
        vectorizer = TfidfVectorizer(
            lowercase=False,
            tokenizer=feat_to_tokens
        )
        X_train_tfidf = vectorizer.fit_transform(pfs)
        svd = TruncatedSVD(self.cnf.n_vecs)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        X_train_lsa = lsa.fit_transform(X_train_tfidf)
        n_neighbors = self.cnf.n_neighbors
        if n_neighbors == -1:
            n_neighbors = len(pfs)
        knn = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights='distance',
            algorithm='brute',
            metric='cosine')
        logger.info(f'training docs: {len(pfs)}, ' +
                    f'labels: {len(Y_train)}')
        knn.fit(X_train_lsa, Y_train)
        model = {'lsa': lsa,
                 'knn': knn,
                 'vectorizer': vectorizer}
        return model

    def predict(self, queries):
        tnfac = TokenNormalizerFactory(self.config)
        tn = tnfac.instance('corpus')
        lr = self.dataloader.lang_res
        tokenizer = lr.tokenizer

        class Query(object):
            def __init__(self, query):
                feats = tuple(lr.features(tokenizer(query), tn))
                self.str_feats = tuple(map(lambda f: f.string_features, feats))
                self.feats = tuple(map(lambda f: f.features, feats))

        feats = map(lambda q: Query(q), queries)
        return self.predict_docs(feats)

    def predict_docs(self, feats):
        model = self.model
        lsa = model['lsa']
        knn = model['knn']
        vectorizer = model['vectorizer']
        X_test_tfidf = vectorizer.transform(feats)
        X_test_lsa = lsa.transform(X_test_tfidf)
        pred_probs = knn.predict_proba(X_test_lsa)
        para_by_id = {p.id: p for p in self.paragraph_train_features}
        preds = []
        for pred_prob in pred_probs:
            ppreds = []
            for i, prob in enumerate(pred_prob):
                if prob > 0:
                    para = para_by_id[knn.classes_[i]]
                    ppreds.append({'paragraph': para,
                                   'probability': prob})
            ppreds.sort(key=lambda x: x['probability'], reverse=True)
            preds.append(ppreds)
        return preds

    def print_query(self, queries=None):
        pstash = self.dataloader.paragraph_stash
        if queries is None:
            queries = ['super bowl', 'sun life stadium']
            #queries = ['super bowl football mandolin']
        preds = self.predict(queries)
        for p in preds[0]:
            pid, proba = p['paragraph'].id, p['probability']
            para = pstash[str(pid)]
            print(pid, proba, para.doc)

    def evaluate(self):
        test_feats, _ = self.dataloader.get_dataset('train')
        test_feats = test_feats[:200]
        preds = self.predict_docs(map(lambda x: x[1], test_feats))

        def mkrow(pi):
            i, pred = pi[0], pi[1][0]['paragraph']
            tp = test_feats[i]
            pred_match = pred.id == tp[0].id
            # correct id, pred id, pred_match, correct label, correct
            return (tp[0].id, pred.id, pred_match, tp[2], pred_match == tp[2])

        pmat = tuple(map(mkrow, zip(it.count(), preds)))
        pmat = np.array(pmat)
        corrects = pmat[pmat[:, 4] == 1]
        acc = corrects.shape[0] / pmat.shape[0]
        print(f'of {len(test_feats)} instances, accuracy: {acc}')

    @classmethod
    def rank_worker(cls, qids):
        return cls(AppConfig.instance()).rank_questions_by_ids(qids)

    def rank_questions_by_ids(self, qids):
        dm = self
        questions = tuple(map(lambda x: dm.questions_by_id[x], qids))
        return dm.rank_questions(questions)

    def rank_questions(self, questions):
        preds = self.predict_docs(questions)
        ranks = []
        for ques, pred in zip(questions, preds):
            logger.debug(f'gold para: {ques.paragraph.id}')
            rank = -1
            for i, p in enumerate(pred):
                if p['paragraph'].id == ques.paragraph.id:
                    rank = i
                    break
            ranks.append((ques.id, rank))
        return ranks

    def question_id_groups(self):
        def key_groups(keys, n):
            klst = tuple(keys)
            for i in range(0, len(klst), n):
                yield klst[i:i+n]

        question_ids = self.questions_by_id.keys()
        chunks = 30
        return tuple(key_groups(question_ids, chunks))

    @persisted('_rank_multi', cache_global=True)
    def rank_multiproc(self, n_workers=6):
        id_sets = self.question_id_groups()
        pool = Pool(n_workers)
        return pool.map(self.__class__.rank_worker, id_sets)

    @persisted('_ranks')
    def calc_ranks(self):
        with time('ranked questions', logger):
            ranks = self.rank_multiproc()
            logger.info(f'ranked {len(ranks)} question groups')
        return ranks

    def write_mrr(self, writer=sys.stdout):
        ranks = tuple(map(lambda x: x[1], it.chain(*self.calc_ranks())))
        logger.debug(f'rank count: {len(ranks)}')
        arr = np.array(tuple(map(lambda r: 1.0 / (r + 1), ranks)))
        s_rank = np.sum(arr)
        Q = len(ranks)
        logger.debug(f'(1.0 / {Q}) * {s_rank}')
        mrr = (1.0 / Q) * s_rank
        std = np.std(arr)
        writer.write(f'{len(ranks)} questions ranked with mean reciprocal rank (MRR): {mrr}\n')
        writer.write(f'aka correct paragraph is ranked as the {1/mrr} most probable average\n')
        writer.write(f'rank standard deviation: {std}\n')

    def sample_features(self):
        pfs = self.dataloader.get_feature_sets('train')
        pfs = pfs[:3]
        print(tuple(map(feat_to_tokens, pfs)))

    def rank_stats(self):
        pfs = self.dataloader.get_feature_sets('train')
        print('question sum:', sum(map(lambda p: len(p.questions), pfs)))
        ids = tuple(map(lambda q: q.id, it.chain(*map(lambda p: p.questions, pfs))))
        print('n ids:', len(ids))
        ids = set(ids)
        print('n ids (set):', len(ids))
        rank_groups = self.calc_ranks()
        print('n ranks groups:', len(rank_groups))
        ranks = tuple(it.chain(*rank_groups))
        print('n ranks:', len(ranks))
        rank_ques_ids = set(map(lambda x: x[0], ranks))
        missing = ids.difference(rank_ques_ids)
        print('missing ranked questions:', len(missing))
        qbyid = self.questions_by_id
        qid = next(iter(missing))
        print(f'{qid}: {qbyid[qid]}')
        groups = self.question_id_groups()
        print(f'n_groups: {len(groups)}')
        ids = tuple(it.chain(*groups))
        print(f'n_ids: {len(ids)}')

    def tmp(self):
        #self._model.clear()
        #self.print_query()
        #self.evaluate()
        #ranks = self.rank_multiproc()
        #self._rank_multi.clear()
        #self.rank_stats()
        self.write_mrr()
