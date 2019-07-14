import logging
import sys
import math
import numpy as np
import time as tm
import itertools as it
import gc
from multiprocessing import Pool
from zensols.actioncli import (
    persisted,
    Stash,
    DirectoryStash,
)
from zensols.dltools.time import time
from zensols.dlqaclass import (
    AppConfig,
    QAModelManager,
)

logger = logging.getLogger(__name__)


def rank_paragraph_set(ids):
    config = AppConfig.instance()
    ranker = Ranker(config)
    ranker.rank_ids(ids)


class QuestionRankStash(Stash):
    def __init__(self, dm_mng: QAModelManager):
        self.paragraphs = dm_mng.data_loader.get_feature_sets('test')
        questions = it.chain(*map(lambda p: p.questions, self.paragraphs))
        self.questions = {str(q.id): q for q in questions}
        self.dm_mng = dm_mng

    @property
    @persisted('_model', cache_global=True)
    def model(self):
        return self.dm_mng.load_model()

    def load(self, qid: str):
        ques = self.questions[qid]
        return self.dm_mng.rank_question(self.model, ques, self.paragraphs)

    @persisted('_keys')
    def keys(self):
        return tuple(map(lambda id: str(id), self.questions.keys()))


class Ranker(object):
    SECTION = 'ranker'

    def __init__(self, config: AppConfig):
        self.config = config
        self.dm_mng = QAModelManager(self.config)

    @property
    def results_path(self):
        return self.config.get_option_path('results_path', self.SECTION)

    @property
    @persisted('_data_loader')
    def data_loader(self):
        return self.dm_mng.data_loader

    @property
    @persisted('_rank_stash', cache_global=True)
    def rank_stash(self):
        return QuestionRankStash(self.dm_mng)

    @property
    @persisted('_results_stash')
    def results_stash(self):
        return DirectoryStash(self.results_path)

    def rank_ids(self, ids):
        results_stash = self.results_stash
        gc_rate = -1
        for i, id in enumerate(ids):
            id = str(id)
            if id in results_stash:
                logger.info(f'already exists: {id}')
            else:
                rank = self.rank_stash.load(id)
                logger.info(f'dumping {id}')
                results_stash.dump(id, rank)
            if ((i % gc_rate) == 0):
                gc.collect()
        return len(ids)

    def compute_ranks(self, n_workers=5):
        stash = self.rank_stash
        n_items = len(stash)
        n_groups = math.ceil(n_items / n_workers)
        id_sets = stash.key_groups(n_groups)
        pool = Pool(n_workers)
        pool.map(rank_paragraph_set, id_sets)

    def calculate_metrics(self, writer=sys.stdout):
        with time('calculated', logger):
            results_stash = self.results_stash
            rank_res = tuple(map(lambda x: x[1], results_stash))
            logger.info(f'calculated or loaeded {len(rank_res)} questions')
        ranked_paragraphs = rank_res[0]['preds'].shape[0]
        logger.info(f"ranked among {ranked_paragraphs} paragraphs")
        #s_rank = sum(map(lambda r: 1.0 / (r['rank'] + 1), rank_res))
        arr = np.array(tuple(map(lambda r: 1.0 / (r['rank'] + 1), rank_res)))
        s_rank = np.sum(arr)
        Q = len(rank_res)
        mrr = (1.0 / Q) * s_rank
        logger.debug(f'(1.0 / {Q}) * {s_rank}')
        return {'mrr': mrr,
                'std': np.std(arr),
                'n_rank_max': ranked_paragraphs,
                'n_paragraphs': Q}

    def write_mrr(self, writer=sys.stdout):
        metrics = self.calculate_metrics()
        mrr, n_paras = metrics['mrr'], metrics['n_paragraphs']
        writer.write(f"""\
{n_paras} questions ranked with mean reciprocal rank (MRR): {mrr:.4f}, or:
correct paragraph is ranked as the {1/mrr:.3f} most probable average
standard deviation: {metrics['std']}
""")

    def paragraph_ids_from_rank(self, rank, top_n=None):
        top_n = 10 if top_n is None else top_n
        return map(lambda i: str(int(i)), rank['preds'][0:top_n, 3])

    def write_ranking(self, rank, top_n=None, writer=sys.stdout):
        ques_id = rank['question_id']
        writer.write(f"question {ques_id} ranks at {rank['rank']}\n")
        gold_para = self.data_loader.paragraph_by_question_id(int(ques_id))
        ques = gold_para.questions_by_key[int(ques_id)]
        writer.write(f'question: {ques.doc}\n')
        writer.write(f'gold paragraph ({gold_para.id}):\n{gold_para.doc}\n')
        for i, para_id in enumerate(self.paragraph_ids_from_rank(rank, top_n)):
            para = self.data_loader.paragraph_stash[para_id]
            writer.write(f'{"-" * 40}\nrank {i} ({para.id}):\n{para.doc}\n')

    def write_ranking_by_id(self, ques_id, top_n=None):
        stash = self.rank_stash
        self.write_ranking(stash[ques_id], top_n)

    def write_first_ranking(self, top_n=None):
        stash = self.rank_stash
        self.write_ranking_by_id(next(iter(stash.keys())), top_n)

    @property
    @persisted('_last_metric', cache_global=True)
    def last_metric(self):
        stash = self.results_stash
        return (tm.time(), len(stash))

    def report_progress(self):
        stash = self.results_stash
        t0, cnt = self.last_metric
        sec_elapse = tm.time() - t0
        cur_ranks = len(stash)
        ranks_elapse = cur_ranks - cnt
        ranks_per_sec = ranks_elapse / sec_elapse
        ranks_total = len(self.rank_stash)
        print()
        print(f'sec: {sec_elapse}, ranks: {ranks_elapse}: ' +
              f'ranks/s: {ranks_per_sec}')
        ranks_left = ranks_total - cur_ranks
        # d = r * t
        time_left_sec = ranks_left / ranks_per_sec
        time_left_min = time_left_sec / 60.
        time_left_hour_f = time_left_min / 60.
        time_left_hour = math.floor(time_left_hour_f)
        time_left_hour_min = (time_left_hour_f - time_left_hour) * 60.0
        print(f'ranks left: {ranks_left}; ({time_left_min:.1f}) ' +
              f'done in {time_left_hour:.0f}:{time_left_hour_min:.0f}')

    def write_ranking_results(self, top_n_ranks=10, top_n_result=2):
        results_stash = self.results_stash
        for _, rres in it.islice(results_stash, top_n_ranks):
            qid = rres['question_id']
            print(f"rank-{qid}: {rres['rank']}")
            self.write_ranking(rres, top_n=top_n_result)
            print('=' * 70)

    def tmp(self):
        self.write_mrr()
