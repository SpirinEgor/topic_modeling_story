import pickle
from collections import Counter
from dataclasses import asdict
from os.path import join
from random import randint
from typing import List

import pyLDAvis as lda_vis
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from numpy import ndarray
import seaborn as sns
from pyLDAvis import PreparedData

from src.dataset import YouTrackIssueDataset

LDA_VIS_DATA = join("models", "lda_vis_data.pkl")
DATA_PATH = join("data", "pycharm_issues.json")
PREDICTION_DATA = join("models", "predictions.pkl")
N_RANDOM_EXAMPLES = 10


def version_counts(dataset: YouTrackIssueDataset) -> Figure:
    version = [it for i in range(len(dataset)) for it in dataset[i][1]]
    all_counts = Counter(version).items()
    sorted_counts = sorted(all_counts, key=lambda x: int(x[0][:4] + x[0][5]))
    versions, counts = zip(*sorted_counts)
    fig = plt.figure(figsize=(20, 5))
    sns.set_theme()
    sns.barplot(x=list(versions), y=list(counts))
    return fig


def retrieve_top_topics_from_version(
    dataset: YouTrackIssueDataset, predictions: ndarray, version: str, top_k: int = 10
) -> List:
    version_indices = [i for i in range(len(dataset)) if version in dataset[i][1]]
    topic = predictions[version_indices].argmax(-1)
    counts = Counter(topic)
    return counts.most_common(top_k)


def get_top_topic_words(topic_id: int, vis_pd: PreparedData, top_k: int = 10) -> List[str]:
    topic_mask = vis_pd.topic_info.Category == f"Topic{topic_id}"
    return vis_pd.topic_info[topic_mask].sort_values(by="Freq", ascending=False).Term[:top_k].tolist()


def show_version(version: str, dataset: YouTrackIssueDataset, prediction: ndarray, vis_pd: PreparedData):
    top_topics = retrieve_top_topics_from_version(dataset, prediction, version)
    top_topics_words = "\n".join(
        [
            f"Topic {topic_id + 1} ({topic_cnt}): " + ", ".join(get_top_topic_words(topic_id + 1, vis_pd))
            for topic_id, topic_cnt in top_topics
        ]
    )
    st.markdown(f"Top-10 most common topics for issues for `{version}` version")
    st.text(top_topics_words)


def main():
    dataset = YouTrackIssueDataset(DATA_PATH)
    with open(PREDICTION_DATA, "rb") as f:
        prediction = pickle.load(f)
    with open(LDA_VIS_DATA, "rb") as f:
        lda_vis_data = pickle.load(f)
    vis_pd = lda_vis.prepare(**lda_vis_data)

    st.set_page_config(layout="wide")
    st.title("Topic modeling on YouTrack issues from PyCharm users.")

    st.markdown(
        """
    Our task is to build topic modeling on issues from YouTrack that were submitted by PyCharm users.

    This streamlit application explains suggested a solution.
    Full code, including model training and all weights, may be found on [GitHub](https://github.com/SpirinEgor/topic_modeling_story).
    """
    )

    st.header("Data overview")
    st.write(
        """
    At first, let's look on an example of issue:
    """.strip()
    )

    st.json(asdict(dataset._issues[0]))

    st.markdown(
        """
    We can see that each issue may correspond to one or mode IDE versions.
    Also, the issue is described by its summary and extensive description.
    In most cases, the description contains snippets with source code or stacktrace.
    A simple text model would struggle with analyzing such data, therefore in this MVP, we will use only a summary.

    Some summary examples:
    """
    )

    random_examples = [f"{i + 1}. {dataset[randint(0, len(dataset) - 1)][0]}" for i in range(N_RANDOM_EXAMPLES)]
    st.text("\n".join(random_examples))

    st.header("Contextualized Topic Modeling")

    st.markdown(
        """
    For building topic modeling we will use contextualized embeddings.
    This approach was presented at ACL'21 and describes
     how we can improve topic modeling by incorporating embeddings from sentence BERT.
    Authors provide easy to usage pip package for model reuse.

    ACL anthology [link](https://aclanthology.org/2021.acl-short.96/)

    GitHub repository [link](https://github.com/MilaNLProc/contextualized-topic-models)

    I trained model on provided data on Google Colab and store its weights, prediction and other necessary data.
    You may refer to [Jupyter Notebook](https://github.com/SpirinEgor/topic_modeling_story/blob/master/train_ctm.ipynb)
    to study training process and download pretrained model from [S3](https://voudy-data.s3.eu-north-1.amazonaws.com/cmt_model.pkl).

    For now, let's focus on built topics based on issue summaries:
    """
    )

    components.html(lda_vis.prepared_data_to_html(vis_pd, template_type="simple"), height=900, scrolling=True)

    st.markdown(
        """
    As we can see, the model coped to detect patterns in incoming issues.
    For example, topic 1 clearly dedicated to a problem with using docker containers,
    and topic 66 is about remote interpreter.
    """
    )

    st.header("Issue per version study.")

    st.markdown(
        """
    Issues may appear in different versions of IDE.
    Therefore it's important to track topics that are related to the concrete version,
    it will allow understanding whether a bug is really fixed with the next version release.

    Let's look into issue distribution per PyCharm version:
    """
    )

    st.pyplot(version_counts(dataset))

    st.markdown(
        """
    Nice to see that the last versions contain less and less issues 🤓.

    Let's focus on topics in the pre-assigned versions, i.e. `2020.2` and `2020.3`.
    """
    )

    show_version("2020.2", dataset, prediction, vis_pd)
    show_version("2020.3", dataset, prediction, vis_pd)

    st.markdown(
        """
    We can see that with moving to the next version issues related to markdown colorscheme, long skeleton updates,
    pep8 style reformat, and so on.
    But also new version brings bugs with plugin updates,
    running remote interpreter on windows, and docker compose support.
    """
    )

    st.header("Trust of validity")
    st.markdown(
        """
    In this MVP multiple assumptions were made:

    1. Number of topics was selected manually through a couple of runs of the model and visual evaluation.
    `100` may not be the best choice and need to be tuned in the future.

    2. As BERT-backend for provided contextual topic modeling Distill Roberta is used.
    This model was trained on corpora of English texts and then distilled in a smaller version.
    Issues contents may be embedded in a close latent space since its one of many domains that the model saw during training.
    Most likely, fine-tuning or training from the scratch model on programming texts would increase model quality.

    3. Description contains much useful information that was skipped in this MVP.
    The main reason for that is code snippets inside it.
    We can detect it and build embeddings with code models, e.g. code2seq, this will bring total embeddings more information.
    """
    )


if __name__ == "__main__":
    main()
