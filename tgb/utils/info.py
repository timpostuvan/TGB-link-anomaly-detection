import os.path as osp
import os

r"""
General space to store global information used elsewhere such as url links, evaluation metrics etc.
"""
PROJ_DIR = osp.dirname(osp.abspath(os.path.join(__file__, os.pardir))) + "/"


class BColors:
    """
    A class to change the colors of the strings.
    """

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


DATA_URL_DICT = {
    "tgbl-wiki": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki-v2.zip",  # "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-wiki.zip", #v1
    "tgbl-review": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review-v2.zip",  # "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-review.zip", #v1
    "tgbl-coin": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-coin.zip",
    "tgbl-flight": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-flight.zip",
    "tgbl-comment": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbl-comment.zip",
    "tgbn-trade": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-trade.zip",
    "tgbn-genre": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-genre.zip",
    "tgbn-reddit": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-reddit.zip",
    "tgbn-token": "https://object-arbutus.cloud.computecanada.ca/tgb/tgbn-token.zip",
    "wikipedia": "https://zenodo.org/record/7213796/files/wikipedia.zip",
    "reddit": "https://zenodo.org/record/7213796/files/reddit.zip",
    "mooc": "https://zenodo.org/record/7213796/files/mooc.zip",
    "lastfm": "https://zenodo.org/record/7213796/files/lastfm.zip",
    "enron": "https://zenodo.org/record/7213796/files/enron.zip",
    "SocialEvo": "https://zenodo.org/record/7213796/files/SocialEvo.zip",
    "uci": "https://zenodo.org/record/7213796/files/uci.zip",
}

DATA_VERSION_DICT = {
    "tgbl-wiki": 2,
    "tgbl-review": 2,
    "tgbl-coin": 1,
    "tgbl-comment": 1,
    "tgbl-flight": 1,
    "tgbn-trade": 1,
    "tgbn-genre": 1,
    "tgbn-reddit": 1,
    "tgbn-token": 1,
    "wikipedia": 1,
    "reddit": 1,
    "mooc": 1,
    "lastfm": 1,
    "enron": 1,
    "SocialEvo": 1,
    "uci": 1,
    "synthetic": 1,
    "lanl": 1,
    "darpa-trace": 1,
    "darpa-theia": 1,
}

DATA_ORGANIC_ANOMALIES = ["lanl", "darpa-trace", "darpa-theia"]

# "https://object-arbutus.cloud.computecanada.ca/tgb/wiki_neg.zip" #for all negative samples of the wiki dataset
# "https://object-arbutus.cloud.computecanada.ca/tgb/review_ns100.zip" #for 100 ns samples in review

DATA_EVAL_METRIC_DICT = {
    "tgbl-wiki": "mrr",
    "tgbl-review": "mrr",
    "tgbl-coin": "mrr",
    "tgbl-comment": "mrr",
    "tgbl-flight": "mrr",
    "tgbn-trade": "ndcg",
    "tgbn-genre": "ndcg",
    "tgbn-reddit": "ndcg",
    "tgbn-token": "ndcg",
}


DATA_NUM_CLASSES = {
    "tgbn-trade": 255,
    "tgbn-genre": 513,
    "tgbn-reddit": 698,
    "tgbn-token": 1001,
}


ANOMALY_ABBREVIATIONS = {
    "temporal-structural-contextual": "tsc",
    "structural-contextual": "sc",
    "temporal-contextual": "tc",
    "temporal": "t",
    "contextual": "c",
    "organic": "organic",
    "combination": "combination",
}
