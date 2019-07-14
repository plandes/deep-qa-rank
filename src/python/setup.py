from pathlib import Path
from zensols.pybuild import SetupUtil

SetupUtil(
    setup_path=Path(__file__).parent.absolute(),
    name="zensols.dlqaclass",
    package_names=['zensols', 'resources'],
    # package_data={'': ['*.html', '*.js', '*.css', '*.map', '*.svg']},
    description='This is a deep learning implementation of question/answer for information retrieval that implements the this paper.',
    user='plandes',
    project='dlqa',
    keywords=['tooling'],
    # has_entry_points=False,
).setup()
