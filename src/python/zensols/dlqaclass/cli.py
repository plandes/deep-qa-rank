import logging
from zensols.actioncli import OneConfPerActionOptionsCliEnv
import zensols.dlqaclass.lsa
from zensols.dlqaclass import (
    AppConfig,
    QADataLoader,
    QAModelManager,
    Ranker,
)


class CorpusExec(object):
    def __init__(self, config):
        self.config = config
        self.dl = QADataLoader(self.config)
        import random
        random.seed(0)

    def reparse_corpus(self):
        self.dl.reparse(force=True, n_workers=10)

    def write_features(self):
        self.dl.write_features(force=True, n_workers=20)


class ModelExec(object):
    def __init__(self, config):
        self.docmng = QAModelManager(config)

    def train(self):
        self.docmng.train()

    def test(self):
        self.docmng.test(self.docmng.load_model())

    def train_test(self):
        self.train()
        self.test()


class RankExec(object):
    def __init__(self, config):
        self.ranker = Ranker(config)

    def compute_ranks(self):
        self.ranker.compute_ranks()

    def calculate_mrr(self):
        self.ranker.write_mrr()


class LsaExec(object):
    def __init__(self, config):
        self.mmng = zensols.dlqaclass.lsa.DocModelManager(config)

    def calculate_mrr(self):
        self.mmng.model
        self.mmng.write_mrr()


class ConfAppCommandLine(OneConfPerActionOptionsCliEnv):
    def __init__(self):
        cnf = {'executors':
               [{'name': 'exporter',
                 'executor': lambda params: CorpusExec(**params),
                 'actions': [{'name': 'parse',
                              'meth': 'reparse_corpus',
                              'doc': '(re)parse the corpus'},
                             {'name': 'features',
                              'meth': 'write_features',
                              'doc': 'write generated features to disk'}]},
                {'name': 'model',
                 'executor': lambda params: ModelExec(**params),
                 'actions': [{'name': 'train',
                              'doc': 'train the model'},
                             {'name': 'test',
                              'doc': 'test the model'},
                             {'name': 'traintest',
                              'meth': 'train_test',
                              'doc': 'train and test the model'}]},
                {'name': 'rank',
                 'executor': lambda params: RankExec(**params),
                 'actions': [{'name': 'calcrank',
                              'meth': 'compute_ranks',
                              'doc': 'rank all questions across all paragraphs'},
                             {'name': 'calcmrr',
                              'meth': 'calculate_mrr',
                              'doc': 'calculate the MRR based on computed ranks'}]},
                {'name': 'lsa',
                 'executor': lambda params: LsaExec(**params),
                 'actions': [{'name': 'lsammr',
                              'meth': 'calculate_mrr',
                              'doc': 'calculate the baseline LSA MRR'}]}],
               'config_option': {'name': 'config',
                                 'expect': True,
                                 'opt': ['-c', '--config', False,
                                         {'dest': 'config',
                                          'metavar': 'FILE',
                                          'help': 'configuration file'}]},
               'whine': 1}
        super(ConfAppCommandLine, self).__init__(
            cnf, config_env_name='dlqarc', pkg_dist='zensols.dlqaclass',
            no_os_environ=True, config_type=AppConfig)

    def _config_logging(self, level):
        root = logging.getLogger()
        map(root.removeHandler, root.handlers[:])
        if level == 0:
            levelno = logging.WARNING
        elif level == 1:
            levelno = logging.INFO
        elif level == 2:
            levelno = logging.DEBUG
        fmt = '%(levelname)s:%(asctime)-15s %(name)s: %(message)s'
        logging.basicConfig(format=fmt, level=levelno)
        root.setLevel(levelno)
        logging.getLogger('zensols.actioncli').setLevel(level=logging.WARNING)


def main():
    cl = ConfAppCommandLine()
    cl.invoke()
