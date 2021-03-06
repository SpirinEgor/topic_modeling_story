import pickle
import time
from typing import Optional

from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from nltk.corpus import stopwords

from src.dataset import YouTrackIssueDataset


class ContextualizedTopicModeling:
    def __init__(self, dataset: YouTrackIssueDataset, model: str):
        documents = [dataset[i][0] for i in range(len(dataset))]
        print(f"Found {len(documents)} documents")

        print("Preprocessing documents...")
        start_time = time.time()
        self._sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords.words("english"))
        self.documents, self.raw_corpus, self.vocab = self._sp.preprocess()
        print(f"Done in {time.time() - start_time} s")

        self.tp = TopicModelDataPreparation(model)
        self.train_dataset = self.tp.fit(text_for_contextual=self.raw_corpus, text_for_bow=self.documents)

        self._model: Optional[ZeroShotTM] = None

    def fit(self, contextual_size: int, n_topics: int, num_epochs: int):
        self._model = ZeroShotTM(
            bow_size=len(self.tp.vocab), contextual_size=contextual_size, n_components=n_topics, num_epochs=num_epochs
        )
        self._model.fit(self.train_dataset)

    @property
    def model(self) -> ZeroShotTM:
        return self._model

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ContextualizedTopicModeling":
        with open(path, "rb") as f:
            return pickle.load(f)
