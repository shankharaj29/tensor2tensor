from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import wiki_lm
from tensor2tensor.utils import registry


_ENDE_TRAIN_DATASETS = [
    [
        "http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz",  # pylint: disable=line-too-long
        ("training-parallel-nc-v13/news-commentary-v13.de-en.en",
         "training-parallel-nc-v13/news-commentary-v13.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz",
        ("commoncrawl.de-en.en", "commoncrawl.de-en.de")
    ],
    [
        "http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz",
        ("training/europarl-v7.de-en.en", "training/europarl-v7.de-en.de")
    ],
]
_ENDE_EVAL_DATASETS = [
    [
        "http://data.statmt.org/wmt17/translation-task/dev.tgz",
        ("dev/newstest2013.en", "dev/newstest2013.de")
    ],
]

@registry.register_problem
class abc(text_problems.Text2TextProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def approx_vocab_size(self):
    return 2**13  # 8192

  @property
  def additional_training_datasets(self):
    """Allow subclasses to add training datasets."""
    return []

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    train_datasets = _ENDE_TRAIN_DATASETS
    return train_datasets if train else _ENDE_EVAL_DATASETS
