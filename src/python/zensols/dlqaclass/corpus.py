import logging
import sys
import itertools as it
from multiprocessing import Pool
import random as rand
import numpy as np
from zensols.actioncli import (
    persisted,
    FactoryStash,
    PersistedWork,
)
from zensols.dltools.time import time
from zensols.nlp import (
    LanguageResourceFactory,
    TokenNormalizerFactory,
    Word2VecModelFactory,
)
from zensols.dlqaclass import (
    AppConfig,
    CorpusReaderFactory,
    ParagraphFeatures,
    ParagraphDictionaryStash,
    Paragraph,
    ParagraphStash,
    ParagraphTokenPopulateStash,
)

logger = logging.getLogger(__name__)


def parse_paragraph_set(ids):
    import gc
    loader = QADataLoader(AppConfig.instance())
    para_stash = loader.paragraph_stash
    for i, id in enumerate(ids):
        id = str(id)
        v = para_stash.load(id)
        para_stash.dump(id, v)
        logger.info(f'parsing: {v}')
        if ((i % 50) == 0):
            gc.collect()
    return len(ids)


def write_feature_sets(ids):
    import gc
    loader = QADataLoader(AppConfig.instance())
    te = loader.token_extractor
    in_stash = loader.paragraph_stash
    out_stash = loader.write_feature_stash
    # out_stash.clear()
    for i, id in enumerate(ids):
        id = str(id)
        para = in_stash[id]
        f = ParagraphFeatures(para, te, None, None)
        logger.info(f'featurizing: {f}')
        out_stash.dump(id, f)
        if ((i % 50) == 0):
            gc.collect()
    return len(ids)


class QADataLoader(object):
    SECTION = 'corpus'

    def __init__(self, config):
        self.config = config
        fac = CorpusReaderFactory(config)
        datasets_name = config.get_option_list('all_readers', self.SECTION)
        self.readers = tuple(map(lambda x: fac.instance(x), datasets_name))
        logger.debug(f'readers: {self.readers}')
        path = self.config.get_option_path('ques_para_map_path', self.SECTION)
        self._ques_to_para = PersistedWork(path, self, True)

    @property
    @persisted('_lang_res')
    def lang_res(self):
        lang_res_fac = LanguageResourceFactory(self.config)
        lr = lang_res_fac.instance()
        return lr

    @property
    @persisted('_word2vec')
    def word2vec(self):
        w2vfac = Word2VecModelFactory(self.config)
        return w2vfac.instance('gensim_goog')

    @property
    @persisted('_te')
    def token_extractor(self):
        tnfac = TokenNormalizerFactory(self.config)
        return tnfac.instance('corpus')

    @property
    def paragraphs(self):
        paras = []
        lr = self.lang_res
        word_embed = self.word2vec
        pid = 0
        gid = 0
        logger.debug(f'create paragraphs with {self.readers}')
        for reader in self.readers:
            ds_name = reader.dataset_name
            logger.info(f'loading and parsing {reader}')
            for pname in reader.get_paragraph_groups():
                for para in reader.paragraphs_by_group(pname):
                    par = Paragraph(pid, gid, ds_name, para, word_embed, lr)
                    paras.append(par)
                    pid += 1
                gid += 1
        logger.info(f'parsed {len(paras)} paragraphs')
        return paras

    @property
    @persisted('_paragraph_stash')
    def paragraph_stash(self):
        path = self.config.get_option_path('lang_parse_path', self.SECTION)
        dir_stash = ParagraphDictionaryStash(self.lang_res, self.word2vec, path)
        return FactoryStash(dir_stash, ParagraphStash(lambda: self.paragraphs))

    @property
    def write_feature_stash(self):
        path = self.config.get_option_path('lang_feature_path', self.SECTION)
        return ParagraphDictionaryStash(self.lang_res, self.word2vec, path)

    @property
    @persisted('_feature_stash', cache_global=True)
    def feature_stash(self):
        stash = ParagraphTokenPopulateStash(self.write_feature_stash)
        stash.post_init()
        stash.set_caching(True)
        stash.set_features(
            self.config.get_option_list('features', self.SECTION))
        return stash

    def get_feature_sets(self, ds_name):
        with time(f'dataset: {ds_name}', logger):
            stash = self.feature_stash
            paras = []
            for id, f in stash:
                # logger.debug(f'{f.ds_name} == {ds_name}')
                if f.ds_name == ds_name:
                    paras.append(f)
            logger.debug(f'loaded {len(stash)}/{len(paras)}')
            return paras

    @property
    @persisted('_ques_to_para')
    def question_to_paragraph_ids(self):
        with time('create question to paragraph mapping', logger):
            ques_to_para = {}
            for _, para in self.feature_stash:
                for q in para.questions:
                    ques_to_para[q.id] = para.id
            return ques_to_para

    def paragraph_by_question_id(self, question_id):
        para_id = self.question_to_paragraph_ids[question_id]
        return self.paragraph_stash[str(para_id)]

    def reparse(self, n_workers=40, force=True, limit=sys.maxsize):
        if force:
            self.write_feature_stash.clear()
        plen = len(self.paragraphs)
        id_sets = np.array_split(np.array(range(plen)), n_workers)
        id_sets = id_sets[:limit]
        pool = Pool(n_workers)
        print(pool.map(parse_paragraph_set, id_sets))

    def write_features(self, n_workers=40, force=True):
        if force:
            self.write_feature_stash.clear()
        plen = len(self.paragraphs)
        id_sets = np.array_split(np.array(range(plen)), n_workers)
        pool = Pool(n_workers)
        print(pool.map(write_feature_sets, id_sets))

    def _create_dataset(self, ds_name, n_neg_ratio=1.0):
        paras = self.get_feature_sets(ds_name)
        total_para = len(paras)
        logger.debug(f'{ds_name} feature sets: {len(paras)}, ' +
                     f'total para: {total_para}')
        questions = list(it.chain(*map(lambda p: p.questions, paras)))
        n_neg_questions = int(len(questions) / total_para * n_neg_ratio)
        logger.debug(f'total question: {len(questions)} ' +
                     f'w/neg ques: {n_neg_questions}')
        rand.shuffle(questions)
        questions = it.cycle(questions)
        pos = []
        neg = []
        for para in paras:
            ques = []
            ids = set()
            while len(ques) < n_neg_questions:
                q = next(questions)
                if q.paragraph == para:
                    logger.debug('question from same paragraph--skipping')
                else:
                    if q.id in ids:
                        logger.warning('adding duplicate question {q.id}')
                    ques.append(q)
                    ids.add(q.id)
            for q in ques:
                neg.append((para, q, False))
            for q in para.questions:
                pos.append((para, q, True))
        logger.debug(f'created pos/neg dataset: {len(pos)}, {len(neg)}')
        return pos, neg

    def get_dataset(self, ds_name, train_split=1.0):
        logger.debug(f'get dataset <{ds_name}> with train split: {train_split}')
        pos, neg = self._create_dataset(ds_name)
        if len(pos) == 0 or len(neg) == 0:
            raise ValueError(f'found +/- {len(pos)}/{len(neg)} instances--' +
                             'features parsed with `write_features` yet?')
        logger.debug(f'dataset size: {len(pos) + len(neg)}')
        pos_idx = int(len(pos) * train_split)
        neg_idx = int(len(neg) * train_split)
        train = pos[:pos_idx] + neg[:neg_idx]
        test = pos[pos_idx:] + neg[neg_idx:]
        rand.shuffle(train)
        rand.shuffle(test)
        logger.info(f'dataset: <{ds_name}> ' +
                    f'train: {len(train)}, test: {len(test)}, total: ' +
                    f'pos: {len(pos)}, neg: {len(neg)}, ' +
                    f'total: {len(pos) + len(neg)}=={len(train) + len(test)}')
        return train, test

    def corpus_stats(self):
        from pprint import pprint
        rds = {str(r): r.stats() for r in self.readers}
        pprint(rds)

    def stats(self):
        train = self.get_dataset('train')[0]
        test = self.get_dataset('test')[0]
        train_pos = sum(1 for _ in filter(lambda x: x[2] is True, train))
        train_neg = sum(1 for _ in filter(lambda x: x[2] is False, train))
        test_pos = sum(1 for _ in filter(lambda x: x[2] is True, test))
        test_neg = sum(1 for _ in filter(lambda x: x[2] is False, test))
        print(f'train: positive={train_pos}, negative={train_neg}')
        print(f'test: positive={test_pos}, negative={test_neg}')
        print(f'totals: train: {len(train)}, test: {len(test)}')

    def tok_count_stats(self):
        stash = self.feature_stash
        para = next(iter(stash))[1]
        print(f'token counts: paragraph: {para.tok_count}, ' +
              f'question: {para.questions[0].tok_count}')

    def max_tokens(self, dses):
        max_para, max_ques = 0, 0
        for ds in dses:
            for para, ques, lab in ds:
                max_para = max(max_para, len(para.feats))
                max_ques = max(max_ques, len(ques.feats))
        logger.info(f'max paragraph: {max_para}, question: {max_ques}')
        return max_para, max_ques

    def tmp(self):
        #self.reparse()
        #self.write_features()
        #self.corpus_stats()
        #self.stats()
        self.tok_count_stats()
