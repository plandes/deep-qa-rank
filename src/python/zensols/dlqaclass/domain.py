import logging
import sys
import pickle
import itertools as it
from functools import reduce
from abc import abstractmethod
import numpy as np
import torch
from spacy.lang.en import TAG_MAP
from zensols.actioncli import (
    persisted,
    PersistableContainer,
    Stash,
    DirectoryStash,
    CacheStash,
    ConfigFactory,
)
from zensols.nlp import (
    TokenNormalizer,
    TokenFeatures,
)

logger = logging.getLogger(__name__)


class MatrixContainer(PersistableContainer):
    def __init__(self, word_embed, lr):
        self.word_embed = word_embed
        self.lr = lr

    def __getstate__(self):
        state = super(MatrixContainer, self).__getstate__()
        for i in 'word_embed lr utterance _mat _feature_mat'.split():
            if i in state:
                del state[i]
        return state

    @property
    @persisted('_doc')
    def doc(self):
        return self.lr.parse(self.utterance)

    def set_tok_count(self, tok_count):
        self._tok_count = tok_count

    @property
    def tok_count(self):
        if not hasattr(self, '_tok_count'):
            raise ValueError(f'no token count set in {self.__class__}')
        return self._tok_count

    @property
    @persisted('_mat', transient=True)
    def matrix(self):
        logger.debug(f'word_embed dim: {self.word_embed.vectors_length}')
        inp_shape = (self.word_embed.vectors_length, self.tok_count,)
        logger.debug(f'para/ques mat: {inp_shape}')
        inp = torch.zeros(inp_shape)
        for col, tok in enumerate(self.doc):
            to = tok.orth_
            if to in self.word_embed:
                vec = self.word_embed[to]
            else:
                vec = self.word_embed['<unk>']
            inp[:, col] = torch.from_numpy(vec)
        return inp

    def __str__(self):
        return f'tokens: {len(self.doc)}: {self.doc[0:5]}'

    def __repr__(self):
        return self.__str__()


class Question(MatrixContainer):
    INST_COUNT = 0

    def __init__(self, paragraph, doc, *args):
        super(Question, self).__init__(*args)
        self.pid = paragraph.id
        self.id = self.__class__.INST_COUNT
        self.__class__.INST_COUNT += 1


class Paragraph(MatrixContainer):
    def __init__(self, id, gid, ds_name, para, word_embed, lr):
        super(Paragraph, self).__init__(word_embed, lr)
        self.id = id
        self.gid = gid
        self.ds_name = ds_name
        self.utterance = para['context']
        self.questions = []
        for qas in para['qas']:
            ques = qas['question']
            if ques.endswith('?'):
                ques = ques[:-1]
            ques_o = Question(self, ques, word_embed, lr)
            ques_o.utterance = ques
            self.questions.append(ques_o)

    @property
    def context(self):
        return self.doc.text

    @property
    @persisted('_questions_by_key', transient=True)
    def questions_by_key(self):
        return {q.id: q for q in self.questions}

    def __setstate__(self, state):
        super(Paragraph, self).__setstate__(state)
        for q in self.questions:
            q.paragraph = self

    def write(self):
        print('question:')
        print(' ', self.doc)
        print('questions:')
        for q in self.questions:
            print(f'  {q.doc}')


class Features(MatrixContainer):
    TAG_TO_ID = {t[0]: t[1] + 1 for t in zip(sorted(TAG_MAP.keys()), it.count())}
    MAX_TAG_ID = max(TAG_TO_ID.values())
    ENTS = 'PERSON NORP FACILITY FAC ORG GPE LOC PRODUCT EVENT WORK_OF_ART LAW LANGUAGE DATE TIME PERCENT MONEY QUANTITY ORDINAL CARDINAL'
    ENTS_TO_ID = {t[0]: t[1] + 1 for t in zip(sorted(ENTS.split()), it.count())}
    MAX_ENTS_ID = max(ENTS_TO_ID.values())
    COMMON_FEATURES_SHAPE = 4
    VEC_FEATS = ()

    def __init__(self, doc, tn: TokenNormalizer, word_embed, lr):
        feats = tuple(map(lambda t: TokenFeatures(doc, *t), tn.normalize(doc)))
        self.str_feats = tuple(map(lambda f: f.string_features, feats))
        self.feats = tuple(map(lambda f: f.features, feats))
        self.word_embed = word_embed
        self.lr = lr

    @property
    def matrix(self):
        if hasattr(self, '_mat'):
            mat = self._mat
        else:
            mat = self._create_matrix()
            if self.cache_matrix:
                self._mat = mat
        return mat

    def _create_matrix(self):
        logger.debug(f'word_embed dim: {self.word_embed.vectors_length}')
        inp_shape = (self.word_embed.vectors_length, self.tok_count)
        logger.debug(f'para/ques mat: {inp_shape}')
        inp = torch.zeros(inp_shape)
        for col, s in enumerate(self.str_feats):
            to = s['norm']
            if to in self.word_embed:
                vec = self.word_embed[to]
            else:
                vec = self.word_embed['<unk>']
            inp[:, col] = torch.from_numpy(vec)
        return inp

    def set_caching(self, matrix):
        self.cache_matrix = matrix

    def _create_feature_matrix(self):
        def map_feat(s, f, n):
            feat = 0
            if n == 'tag':
                t = s['tag']
                if t in self.TAG_TO_ID:
                    feat = self.TAG_TO_ID[t] / self.MAX_TAG_ID
            elif n == 'ent':
                if s['is_entity']:
                    t = s['entity']
                    if t in self.ENTS_TO_ID:
                        feat = self.ENTS_TO_ID[t] / self.MAX_ENTS_ID
            elif n == 'i':
                feat = f[n] / self.tok_count
            else:
                feat = f[n]
            return feat

        n_feats = len(self.features)
        inp_shape = (n_feats, self.tok_count)
        inp = torch.zeros(inp_shape)
        for col, (s, f) in enumerate(zip(self.str_feats, self.feats)):
            feats = np.array(tuple(map(lambda n: map_feat(s, f, n),
                                       self.features)),
                             dtype=np.float32)
            inp[:, col] = torch.from_numpy(feats)
        return inp

    @property
    def feature_matrix(self):
        if hasattr(self, '_feature_mat'):
            feature_mat = self._feature_mat
        else:
            feature_mat = self._create_feature_matrix()
            if self.cache_matrix:
                self._feature_mat = feature_mat
        return feature_mat

    def set_features(self, features):
        self.features = features
        if hasattr(self, '_feature_mat'):
            delattr(self, '_feature_mat')

    @staticmethod
    def words(feat):
        return set(map(lambda x: x[1]['norm'],
                       filter(lambda x: not x[0]['is_stop'],
                              zip(feat.feats, feat.str_feats))))

    @staticmethod
    def ents(feat):
        return set(map(lambda x: x[0]['entity'],
                       filter(lambda x: x[0]['is_entity'],
                              zip(feat.feats, feat.str_feats))))

    def common_matrix(self, ques):
        para = self
        common_words = self.words(para) & self.words(ques)
        common_ents = self.ents(para) & self.ents(ques)
        inp = torch.tensor((len(common_words), len(common_ents),
                            len(para.feats), len(ques.feats)),
                           dtype=torch.float)
        return inp

    def __getstate__(self):
        state = super(Features, self).__getstate__()
        return state

    def write(self, writer=sys.stdout, limit=sys.maxsize):
        from pprint import pprint
        for tf in it.islice(self.str_feats, limit):
            pprint(tf, writer)

    def __str__(self):
        return f'{self.id}: features: {len(self.feats)}'


class QuestionFeatures(Features):
    def __init__(self, question: Question, *args, **kwargs):
        super(QuestionFeatures, self).__init__(question.doc, *args, **kwargs)
        self.id = question.id


class ParagraphFeatures(Features):
    def __init__(self, paragraph: Paragraph, te: TokenNormalizer,
                 word_embed, lr):
        super(ParagraphFeatures, self).__init__(
            paragraph.doc, te, word_embed, lr)
        self.id = paragraph.id
        self.gid = paragraph.gid
        self.ds_name = paragraph.ds_name
        self.questions = tuple(
            map(lambda q: QuestionFeatures(q, te, word_embed, lr),
                paragraph.questions))

    def __setstate__(self, state):
        super(ParagraphFeatures, self).__setstate__(state)
        for q in self.questions:
            q.paragraph = self


class ParagraphStash(Stash):
    def __init__(self, para_fn):
        self.para_fn = para_fn

    @property
    @persisted('_para')
    def paragraphs(self):
        return {str(p.id): p for p in self.para_fn()}

    def load(self, name: str):
        return self.paragraphs[name]

    @persisted('_keys')
    def keys(self):
        return self.paragraphs.keys()


class ParagraphDictionaryStash(DirectoryStash):
    def __init__(self, lr, word_embed, *args, **kwargs):
        super(ParagraphDictionaryStash, self).__init__(*args, **kwargs)
        self.lr = lr
        self.word_embed = word_embed

    def load(self, name):
        if not isinstance(name, str):
            raise ValueError(f'wrong key type: {type(name)}')
        para = super(ParagraphDictionaryStash, self).load(name)
        para.lr = self.lr
        para.word_embed = self.word_embed
        for q in para.questions:
            q.lr = self.lr
            q.word_embed = self.word_embed
        return para

    def dump(self, name, inst):
        logger.info(f'saving instance: {inst}')
        path = self._get_instance_path(name)
        with open(path, 'wb') as f:
            pickle.dump(inst, f)


class ParagraphTokenPopulateStash(CacheStash):
    def __init__(self, delegate):
        super(ParagraphTokenPopulateStash, self).__init__(delegate)

    def post_init(self):
        def reduce_max(md, feat_tup):
            feat = feat_tup[1]
            m_para = len(feat.feats)
            m_ques = max(map(lambda q: len(q.feats), feat.questions))
            md[0] = max(md[0], m_para)
            md[1] = max(md[1], m_ques)
            return md

        para_t_count, ques_t_count = reduce(reduce_max, self, [0, 0])
        for _, para in self:
            para.set_tok_count(para_t_count)
            for q in para.questions:
                q.set_tok_count(ques_t_count)

    def set_caching(self, cache_matrix):
        for _, para in self:
            para.set_caching(cache_matrix)
            for q in para.questions:
                q.set_caching(cache_matrix)

    def set_features(self, features):
        for _, para in self:
            para.set_features(features)
            for q in para.questions:
                q.set_features(features)


class CorpusParserFactory(ConfigFactory):
    INSTANCE_CLASSES = {}

    def __init__(self, config):
        super(CorpusParserFactory, self).__init__(config, '{name}_corpus_parser')


class CorpusReader(object):
    @abstractmethod
    def get_paragraph_groups(self):
        pass

    @abstractmethod
    def paragraphs_by_group(self, name=None):
        pass


class CorpusReaderFactory(ConfigFactory):
    INSTANCE_CLASSES = {}

    def __init__(self, config):
        super(CorpusReaderFactory, self).__init__(config, '{name}_corpus_reader')
