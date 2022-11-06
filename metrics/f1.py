# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" ROUGE metric from Google Research github repo. """

# The dependencies in https://github.com/google-research/google-research/blob/master/rouge/requirements.txt

import datasets
import re


_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.

Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.

This metrics is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge
"""

_KWARGS_DESCRIPTION = """
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
    rouge_types: A list of rouge types to calculate.
        Valid names:
        `"rouge{n}"` (e.g. `"rouge1"`, `"rouge2"`) where: {n} is the n-gram based scoring,
        `"rougeL"`: Longest common subsequence based scoring.
        `"rougeLSum"`: rougeLsum splits text using `"\n"`.
        See details in https://github.com/huggingface/datasets/issues/617
    use_stemmer: Bool indicating whether Porter stemmer should be used to strip word suffixes.
    use_agregator: Return aggregates if this is set to True
Returns:
    rouge1: rouge_1 (precision, recall, f1),
    rouge2: rouge_2 (precision, recall, f1),
    rougeL: rouge_l (precision, recall, f1),
    rougeLsum: rouge_lsum (precision, recall, f1)
Examples:

    >>> rouge = datasets.load_metric('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
    >>> print(results["rouge1"])
    AggregateScore(low=Score(precision=1.0, recall=1.0, fmeasure=1.0), mid=Score(precision=1.0, recall=1.0, fmeasure=1.0), high=Score(precision=1.0, recall=1.0, fmeasure=1.0))
    >>> print(results["rouge1"].mid.fmeasure)
    1.0
"""


def extract_ne_list(labels):
    ne_list = []
    ne_counts = []

    pattern = r'<.*?:.*?>'

    for row in labels:
        items = re.findall(pattern, row)

        ne_items = [item.strip("<>").split(":") for item in items]

        ne_counts.append(len(items))
        ne_list.append(ne_items)

    return ne_list, ne_counts


def compute_f1_score(labels, counts, predictions):
    precision_all, recall_all, f1_all = 0, 0, 0

    for index in range(len(labels)):
        ne_item = labels[index]
        ne_count = counts[index]
        pred_item = predictions[index]

        tp, fp = 0, 0

        for ne in ne_item:
            if ne in pred_item:
                tp += 1

        for pred in pred_item:
            if pred not in ne_item:
                fp += 1

        precision = 0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0 if ne_count == 0 else tp / (ne_count)

        inverse_precision = 0 if precision == 0 else 1 / precision
        inverse_recall = 0 if recall == 0 else 1 / recall
        
        f1 = 0 if precision == 0 and recall == 0 else 2 / (inverse_precision + inverse_recall)

        precision_all += precision
        recall_all += recall
        f1_all += f1

    precision_all /= len(labels)
    recall_all /= len(labels)
    f1_all /= len(labels)

    return {
        "precision": precision_all,
        "recall": recall_all,
        "f1": f1_all
    }


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class F1(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string", id="sequence"),
                    "references": datasets.Value("string", id="sequence"),
                }
            ),
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )

    def _compute(self, predictions, references):
        ne_list, ne_counts = extract_ne_list(references)
        pred_ne_list, _ = extract_ne_list(predictions)

        result = compute_f1_score(ne_list, ne_counts, pred_ne_list)

        return result
