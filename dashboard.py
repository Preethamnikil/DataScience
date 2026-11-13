from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Indian Election Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

# =========================================================
# PROFESSIONAL DARK THEME
# =========================================================

px.defaults.template = "plotly_dark"

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            ellipse at 5% 0%,
            rgba(99, 102, 241, 0.15),
            transparent 35%
        ),
        #0b1020;

    color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}

h1 {
    color: #f8fafc !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2, h3 {
    color: #e2e8f0 !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(
        135deg,
        #1e293b,
        #111827
    );

    border: 1px solid #334155;
    border-radius: 18px;
    padding: 22px;
    min-height: 125px;

    box-shadow: 0 8px 25px rgba(0,0,0,0.25);

    transition: all 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    border-color: #8b5cf6;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #a5b4fc !important;
    font-weight: 600 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #ffffff !important;
    font-size: 30px !important;
    font-weight: 800 !important;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #111827,
        #0b1020
    );

    border-right: 1px solid #334155;
}

section[data-testid="stSidebar"] * {
    color: #e2e8f0;
}

[data-testid="stPlotlyChart"] {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 8px;
}

.stButton button {
    border-radius: 10px;
    font-weight: 650;
}

.stButton button[kind="primary"] {
    background: linear-gradient(
        90deg,
        #7c3aed,
        #4f46e5
    );

    color: white;
    border: none;
}

hr {
    border-color: #334155;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_summary():
    path = RESULTS / "summary.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_metrics():
    return pd.read_csv(RESULTS / "metrics.csv")


@st.cache_data
def load_state_summary():
    path = RESULTS / "state_summary.json"

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_state_tweets():
    path = RESULTS / "state_tweets.csv"

    if path.exists():
        return pd.read_csv(path)

    return pd.DataFrame()


@st.cache_resource
def load_model():
    vectorizer = joblib.load(
        MODELS / "tfidf_vectorizer.joblib"
    )

    model = joblib.load(
        MODELS / "svm.joblib"
    )

    return vectorizer, model


# =========================================================
# CHECK DATA
# =========================================================

try:
    summary = load_summary()
    metrics = load_metrics()

except Exception as e:
    st.error(f"Unable to load dashboard data: {e}")
    st.info(
        "Check whether results/summary.json "
        "and results/metrics.csv exist."
    )
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("# 📊 Sentiment AI")

    st.caption(
        "Indian General Election 2019"
    )

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Sentiment Analysis",
            "Model Performance",
            "State Analysis",
            "Leader Analysis",
            "Tweet Analyzer",
        ],
    )

    st.divider()

    st.markdown(
        """
        **Project Information**

        Dataset: Indian Election Tweets

        Classification:
        Positive / Neutral / Negative

        Best Model: SVM
        """
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    # Sentiment Analysis of Indian Election Tweets

    ### AI-powered analysis of public opinion during Lok Sabha 2019
    """
)

st.caption(
    "2019 Indian General Election • English Tweets • "
    "Machine Learning & NLP"
)

st.divider()


# =========================================================
# OVERVIEW PAGE
# =========================================================

if page == "Overview":

    total = summary["n_labeled_after_cleaning"]
    train = summary["train_size"]
    test = summary["test_size"]

    best_model = summary["best_overall_model"]
    best_f1 = summary["best_overall_f1"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Tweets",
        f"{total:,}"
    )

    c2.metric(
        "Training Tweets",
        f"{train:,}"
    )

    c3.metric(
        "Best Model",
        best_model
    )

    c4.metric(
        "Best F1 Score",
        f"{best_f1 * 100:.2f}%"
    )

    st.markdown("## 📈 Dashboard Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sentiment Distribution")

        counts = summary["label_counts"]

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(counts.keys()),
                    values=list(counts.values()),
                    hole=0.55,
                    marker=dict(
                        colors=[
                            "#fb7185",
                            "#fbbf24",
                            "#34d399",
                        ]
                    ),
                    textinfo="label+percent",
                )
            ]
        )

        fig.update_layout(
            height=380,
            showlegend=True,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("Tweet Count by Sentiment")

        sentiment_df = pd.DataFrame(
            {
                "Sentiment": list(counts.keys()),
                "Tweets": list(counts.values()),
            }
        )

        fig = px.bar(
            sentiment_df,
            x="Sentiment",
            y="Tweets",
            color="Sentiment",
            color_discrete_map={
                "positive": "#34d399",
                "negative": "#fb7185",
                "neutral": "#fbbf24",
            },
            text="Tweets",
        )

        fig.update_layout(
            height=380,
            showlegend=False,
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.markdown("## 🏆 Model Performance")

    st.dataframe(
        metrics.sort_values(
            "f1",
            ascending=False
        ).style.format(
            {
                "accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# SENTIMENT ANALYSIS PAGE
# =========================================================

elif page == "Sentiment Analysis":

    st.markdown("## Sentiment Distribution")

    counts = summary["label_counts"]

    sentiment_df = pd.DataFrame(
        {
            "Sentiment": list(counts.keys()),
            "Count": list(counts.values()),
        }
    )

    fig = px.bar(
        sentiment_df,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map={
            "positive": "#34d399",
            "negative": "#fb7185",
            "neutral": "#fbbf24",
        },
        text="Count",
    )

    fig.update_layout(
        height=500,
        title="Overall Sentiment Analysis",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Sentiment Percentages")

    total = sum(counts.values())

    for sentiment, count in counts.items():

        percentage = count / total * 100

        st.write(
            f"**{sentiment.capitalize()}** — "
            f"{percentage:.2f}%"
        )

        st.progress(
            percentage / 100
        )


# =========================================================
# MODEL PERFORMANCE PAGE
# =========================================================

elif page == "Model Performance":

    st.markdown("## Machine Learning Model Comparison")

    fig = px.bar(
        metrics,
        x="model",
        y="accuracy",
        color="model",
        text="accuracy",
        title="Accuracy Comparison",
    )

    fig.update_layout(
        height=450,
        yaxis=dict(
            range=[0, 1]
        ),
        showlegend=False,
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.line(
        metrics,
        x="model",
        y=["precision", "recall", "f1"],
        markers=True,
        title="Precision, Recall and F1 Score",
    )

    fig2.update_layout(
        height=450,
        yaxis=dict(
            range=[0, 1]
        ),
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("### Detailed Metrics")

    st.dataframe(
        metrics.style.format(
            {
                "accuracy": "{:.2%}",
                "precision": "{:.2%}",
                "recall": "{:.2%}",
                "f1": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# STATE ANALYSIS PAGE
# =========================================================

elif page == "State Analysis":

    st.markdown("## State-wise Sentiment Analysis")

    try:

        state_data = load_state_summary()

        states = state_data["states"]

        state_df = pd.DataFrame.from_dict(
            states,
            orient="index"
        )

        state_df.index.name = "State"

        state_df = state_df.reset_index()

        state_df = state_df.sort_values(
            "tweet_mentions",
            ascending=False
        )

        selected_state = st.selectbox(
            "Select a State",
            state_df["State"].tolist()
        )

        selected = state_df[
            state_df["State"] == selected_state
        ].iloc[0]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Tweet Mentions",
            f"{selected['tweet_mentions']:,}"
        )

        c2.metric(
            "Positive Tweets",
            f"{selected['positive']:,}"
        )

        c3.metric(
            "Negative Tweets",
            f"{selected['negative']:,}"
        )

        st.markdown(
            f"### Sentiment in {selected_state}"
        )

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=[
                        "Positive",
                        "Neutral",
                        "Negative",
                    ],
                    values=[
                        selected["positive"],
                        selected["neutral"],
                        selected["negative"],
                    ],
                    hole=0.55,
                    marker=dict(
                        colors=[
                            "#34d399",
                            "#fbbf24",
                            "#fb7185",
                        ]
                    ),
                )
            ]
        )

        fig.update_layout(
            height=400
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("### State Information")

        st.write(
            f"**Region:** {selected['region']}"
        )

        st.write(
            f"**Lok Sabha Seats:** "
            f"{selected['lok_sabha_seats']}"
        )

        st.write(
            f"**Major Parties:** "
            f"{selected['major_parties']}"
        )

        st.write(
            f"**Election Result:** "
            f"{selected['result_2019']}"
        )

        st.markdown("### Top States by Tweet Mentions")

        fig2 = px.bar(
            state_df.head(10),
            x="State",
            y="tweet_mentions",
            color="tweet_mentions",
            color_continuous_scale="Viridis",
        )

        fig2.update_layout(
            height=450
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"State analysis could not be loaded: {e}"
        )


# =========================================================
# LEADER ANALYSIS PAGE
# =========================================================

elif page == "Leader Analysis":

    st.markdown("## Political Leader Sentiment")

    leader_data = summary["leader_sentiment"]

    rows = []

    for sentiment, leaders in leader_data.items():

        for leader, percentage in leaders.items():

            rows.append(
                {
                    "Leader": leader,
                    "Sentiment": sentiment.capitalize(),
                    "Percentage": percentage * 100,
                }
            )

    leader_df = pd.DataFrame(rows)

    fig = px.bar(
        leader_df,
        x="Leader",
        y="Percentage",
        color="Sentiment",
        barmode="group",
        color_discrete_map={
            "Positive": "#34d399",
            "Negative": "#fb7185",
            "Neutral": "#fbbf24",
        },
        title="Predicted Sentiment by Political Leader",
    )

    fig.update_layout(
        height=450,
        yaxis_title="Percentage",
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("### Leader Tweet Counts")

    leader_counts = summary["leader_raw_counts"]

    st.dataframe(
        pd.DataFrame(
            {
                "Leader": list(leader_counts.keys()),
                "Tweets": list(leader_counts.values()),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# TWEET ANALYZER PAGE
# =========================================================

elif page == "Tweet Analyzer":

    st.markdown("## Live Tweet Sentiment Prediction")

    st.caption(
        "Enter a tweet and let the trained SVM model "
        "predict its sentiment."
    )

    tweet = st.text_area(
        "Enter Tweet",
        placeholder=(
            "Example: The government is doing "
            "a great job for development."
        ),
        height=150,
    )

    if st.button(
        "Analyze Sentiment",
        type="primary",
        use_container_width=True,
    ):

        if not tweet.strip():

            st.warning(
                "Please enter a tweet first."
            )

        else:

            try:

                vectorizer, model = load_model()

                # Use the same preprocessing
                # used during model training.

                from src.preprocess import clean_tweet

                cleaned = clean_tweet(tweet)

                prediction = model.predict(
                    vectorizer.transform(
                        [cleaned]
                    )
                )[0]

                st.divider()

                st.markdown(
                    "### Prediction Result"
                )

                if prediction == "positive":

                    st.success(
                        "😊 Positive Sentiment"
                    )

                elif prediction == "negative":

                    st.error(
                        "😠 Negative Sentiment"
                    )

                else:

                    st.info(
                        "😐 Neutral Sentiment"
                    )

                st.write(
                    "**Original Tweet:**"
                )

                st.write(tweet)

                st.write(
                    "**Cleaned Tweet:**"
                )

                st.code(cleaned)

                st.write(
                    "**Predicted Sentiment:**"
                )

                st.markdown(
                    f"### {prediction.capitalize()}"
                )

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Sentiment Analysis of Indian Election Tweets | "
    "Machine Learning Project | 2019 Lok Sabha Election"
)