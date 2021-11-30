import pickle
from collections import Counter
from dataclasses import asdict
from os.path import join
from random import randint

import plotly.express as px
import pyLDAvis as lda_vis
import streamlit as st
import streamlit.components.v1 as components
from plotly.graph_objs import Figure

from src.dataset import YouTrackIssueDataset

LDA_VIS_DATA = join("models", "vis_data.pkl")
DATA_PATH = join("data", "pycharm_issues.json")
N_RANDOM_EXAMPLES = 10


def build_lda_vis_html() -> str:
    with open(LDA_VIS_DATA, "rb") as f:
        lda_vis_data = pickle.load(f)
    vis_pd = lda_vis.prepare(**lda_vis_data)
    return lda_vis.prepared_data_to_html(vis_pd, template_type="simple")


def version_counts(dataset: YouTrackIssueDataset) -> Figure:
    version = [it for i in range(len(dataset)) for it in dataset[i][1]]
    all_counts = Counter(version).items()
    sorted_counts = sorted(all_counts, key=lambda x: int(x[0][:4] + x[0][5]))
    versions, counts = zip(*sorted_counts)
    fig = px.bar(x=list(versions), y=list(counts))
    return fig


def main():
    dataset = YouTrackIssueDataset(DATA_PATH)
    st.set_page_config(layout="wide")
    st.title("Topic modeling on YouTrack issues from PyCharm users.")

    st.markdown(
        """
    Our task is to build topic modeling on issues from YouTrack that were submitted by PyCharm users.

    This streamlit application explains suggested a solution.
    Full code, including model training and all weights, may be found on [GitHub](https://github.com/SpirinEgor/topic_modeling_story).
    """.strip()
    )

    st.header("Data overview")
    st.write(
        """
    At first, let's look on an example of issue:
    """.strip()
    )

    st.json(asdict(dataset._issues[0]))

    st.write(
        """
    We can see that each issue may correspond to one or mode IDE versions.
    Also, the issue is described by its summary and extensive description.
    In most cases, the description contains snippets with source code or stacktrace.
    A simple text model would struggle with analyzing such data, therefore in this MVP, we will use only a summary.

    Some summary examples:
    """.strip()
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
    to study training process.

    For now, let's focus on built topics based on issue summaries:
    """.strip()
    )

    components.html(build_lda_vis_html(), height=1000, scrolling=True)

    st.write(
        """
    As we can see, the model coped to detect patterns in incoming issues.
    For example, topic X clearly dedicated to a problem with ...,
    and topic Y is about ....
    """.strip()
    )

    st.header("Issue per version study.")

    st.markdown(
        """
    Issues may appear in different versions of IDE.
    Therefore it's important to track topics that are related to the concrete version,
    it will allow understanding whether a bug is really fixed with the next version release.

    Let's look into issue distribution per PyCharm version:
    """.strip()
    )

    st.plotly_chart(version_counts())

    st.markdown(
        """
    Nice to see that the last versions contain less and less issues 🤓.

    Let's focus on topics in the pre-assigned versions.
    At first, we will retrieve top-10 topics in `2020.2` and `2020.3` versions to see most often problems with these releases.
    After that, we can check which topics were solved and which ones appear.
    """.strip()
    )

    st.write("IN PROGRESS 👨‍💻")


if __name__ == "__main__":
    main()
