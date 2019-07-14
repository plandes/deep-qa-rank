import logging
from zensols.actioncli import ClassImporter
from zensols.dlqaclass import AppConfig

logger = logging.getLogger(__name__)


def inst(name, info_loggers='', debug_loggers=''):
    for i in info_loggers.split():
        logging.getLogger(f'zensols.dlqaclass.{i}').setLevel(logging.INFO)
    for i in debug_loggers.split():
        logging.getLogger(f'zensols.dlqaclass.{i}').setLevel(logging.DEBUG)
    config = AppConfig('resources/dlqa.conf')
    return ClassImporter('zensols.dlqaclass.' + name).instance(config)


def corpus():
    inst('corpus.QADataLoader', 'corpus squad').tmp()


def model():
    inst('model.DocModelManager', 'corpus model').tmp()


def rank():
    inst('rank.Ranker', 'model rank').tmp()


def lsa():
    inst('lsa.DocModelManager', '', 'lsa').tmp()


def main():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('zensols.actioncli').setLevel(logging.WARN)
    logging.getLogger('zensols.dlqaclass.app').setLevel(logging.INFO)
    run = 3
    {1: corpus,
     2: model,
     3: rank,
     4: lsa,
     }[run]()


main()
