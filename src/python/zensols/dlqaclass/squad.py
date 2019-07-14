import logging
import json
from pathlib import Path
import urllib.request
import itertools as it
from zensols.actioncli import persisted
from zensols.dlqaclass import (
    CorpusReader,
    CorpusReaderFactory,
    CorpusParserFactory,
)

logger = logging.getLogger(__name__)


class SquadCorpusParser(object):
    def __init__(self, config, version, file_path, url, path_name):
        self.version = version
        params = {'path_name': path_name, 'version': version}
        self.url = url.format(**params)
        self.file_path = Path(file_path.format(**params))

    def _assert_downloaded(self):
        logger.debug(f'asserting downloaded {self.file_path}')
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            logger.info(f'downloading {self.url} -> {self.file_path}')
            urllib.request.urlretrieve(self.url, self.file_path)

    def parse(self):
        self._assert_downloaded()
        logger.debug(f'reading JSON from {self.file_path}')
        with open(self.file_path) as f:
            content = json.load(f)
        return content['data']


CorpusParserFactory.register(SquadCorpusParser)


class SquadCorpusReader(CorpusReader):
    def __init__(self, config, dataset_name, corpus_parser, path_name):
        self.parser = CorpusParserFactory(config).instance(
            corpus_parser, path_name=path_name)
        self.dataset_name = dataset_name
        self.path_name = path_name
        logger.debug(f'created reader: {self}')

    @property
    @persisted('_data')
    def data(self):
        return self.parser.parse()

    def print_titles(self):
        for title in self.get_paragraph_groups():
            print(title)

    def get_paragraph_groups(self):
        return map(lambda x: x['title'], self.data)

    def paragraphs_by_group(self, name):
        for article in self.data:
            if article['title'] == name:
                return article['paragraphs']

    def print_paragraphs(self, name, limit=1):
        for para in it.islice(self.paragraphs_by_group(name), limit):
            context = para['context']
            print(f'context:\n    {context}')
            for qas in para['qas']:
                ques = qas['question']
                print(f'question: {ques}')
                ans_texts = set()
                for ans in qas['answers']:
                    ans_start = ans['answer_start']
                    text = ans['text']
                    ans_texts.add(f'  {text} ({ans_start})')
                for at in ans_texts:
                    print(at)

    def stats(self):
        articles = []
        paragraphs = []
        answers = []
        questions = 0
        question_words = 0
        answer_words = 0
        for title in self.get_paragraph_groups():
            arts = self.paragraphs_by_group(title)
            articles.append((title, arts,))
            for para in arts:
                paragraphs.append(para)
                for qas in para['qas']:
                    ques = qas['question']
                    questions += 1
                    question_words += ques.count(' ') + 1
                    for ans in qas['answers']:
                        text = ans['text']
                        answers.append(ans)
                        answer_words += text.count(' ') + 1
        return {'articles': len(articles),
                'paragraphs': len(paragraphs),
                'questions': questions,
                'answers': len(answers),
                'question_words': question_words,
                'answer_words': answer_words}

    def __str__(self):
        return f'ds: {self.dataset_name}, path: {self.path_name}'

    def __repr__(self):
        return self.__str__()


CorpusReaderFactory.register(SquadCorpusReader)
