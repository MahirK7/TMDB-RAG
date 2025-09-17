import streamlit as st
from rag_chain import build_qa_chain
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub, os

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_path = os.path.join(path, csv_files[0])
    return pd.read_csv(csv_path)

df = load_data()

# ---------------------------
# Sidebar Filters
# ---------------------------
st.sidebar.title("⚙️ Settings")

# Try loading an LLM backend
qa = None
error_msg = None
model_choice = None
try:
    model_choice = st.sidebar.selectbox(
        "Choose an LLM backend",
        ["ollama: llama3:3b", "ollama: mistral:7b", "openai: gpt-4o-mini", "anthropic: claude-3-sonnet"]
    )
    qa = build_qa_chain(model_choice)
except Exception as e:
    error_msg = str(e)

st.sidebar.markdown("---")
st.sidebar.header("📊 Data Filters")

# Year filter
min_year = int(df["first_air_date"].dropna().str[:4].min())
max_year = int(df["first_air_date"].dropna().str[:4].max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# Genre filter
all_genres = sorted(set(df["genres"].dropna().str.split(",").explode()))
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

# Platform filter (based on binary columns)
platforms = ["Netflix", "Hulu", "Prime Video", "Disney+"]
selected_platforms = st.sidebar.multiselect("Select Platforms", platforms)

# ---------------------------
# Apply Filters
# ---------------------------
df_filtered = df.copy()

# Filter by year
df_filtered = df_filtered[
    df_filtered["first_air_date"].str[:4].fillna("0").astype(int).between(year_range[0], year_range[1])
]

# Filter by genres
if selected_genres:
    df_filtered = df_filtered[
        df_filtered["genres"].dropna().apply(lambda g: any(genre in g for genre in selected_genres))
    ]

# Filter by platforms
if selected_platforms:
    mask = df_filtered[selected_platforms].any(axis=1)
    df_filtered = df_filtered[mask]

# ---------------------------
# Tabs: Conditional
# ---------------------------
if qa:
    tab1, tab2 = st.tabs(["🔎 Ask Questions", "📊 Data Analysis"])

    # --- Tab 1: RAG Q&A ---
    with tab1:
        st.header("Ask Questions about TV Shows")
        query = st.text_input("Enter your question:")
        if query:
            result = qa.invoke(query)
            st.write("### Answer")
            st.write(result["result"])
            st.write("### Sources")
            for doc in result["source_documents"]:
                st.write(doc.page_content[:400] + "...")
                st.write("---")

    # --- Tab 2: Seaborn Analysis ---
    with tab2:
        st.header("TV Show Dataset Analysis")

else:
    st.warning("⚠️ No LLM backend available. Showing only data analysis.")
    st.header("📊 TV Show Dataset Analysis")

# ---------------------------
# Shared Seaborn Analysis (uses df_filtered)
# ---------------------------

# Release Year Trend
st.subheader("📈 Release Year Trend (Top 20)")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(
    data=df_filtered,
    x="first_air_date",
    order=df_filtered["first_air_date"].value_counts().index[:20],
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# Genre Distribution
st.subheader("🎭 Top 15 Genres")
fig, ax = plt.subplots(figsize=(10, 6))
df_filtered["genres"].dropna().str.split(",").explode().value_counts()[:15].plot(
    kind="bar", color="skyblue", ax=ax
)
st.pyplot(fig)

# Rating Distribution
st.subheader("⭐ Rating Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_filtered["vote_average"], bins=20, kde=True, color="purple", ax=ax)
st.pyplot(fig)

# Country Distribution
st.subheader("🌍 Top 10 Producing Countries")
fig, ax = plt.subplots(figsize=(12, 6))
df_filtered["origin_country"].dropna().str.split(",").explode().value_counts()[:10].plot(
    kind="bar", color="orange", ax=ax
)
st.pyplot(fig)
