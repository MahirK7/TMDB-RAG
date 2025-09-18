# Import core libraries
import streamlit as st                     # Streamlit for building the web app
from rag_chain import build_qa_chain       # Custom function to build RAG Q&A chain (your own module)
import pandas as pd                        # For dataset handling
import seaborn as sns                      # For plotting/visualizations
import matplotlib.pyplot as plt            # For plotting (works with seaborn)
import kagglehub, os                       # For dataset download + file handling


# ---------------------------
# Load dataset
# ---------------------------

# Cache the dataset loading so it's only downloaded/read once (improves performance in Streamlit)
@st.cache_data
def load_data():
    # Download dataset from Kaggle (returns the local folder path)
    path = kagglehub.dataset_download("asaniczka/full-tmdb-tv-shows-dataset-2023-150k-shows")
    # Find the CSV file in that folder
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    csv_path = os.path.join(path, csv_files[0])
    # Load CSV into pandas DataFrame
    return pd.read_csv(csv_path)

# Actually load the dataset
df = load_data()


# ---------------------------
# Sidebar Filters
# ---------------------------

st.sidebar.title("⚙️ Settings")  # Sidebar section title

# Try to load an LLM backend for Q&A
qa = None
error_msg = None
model_choice = None
try:
    # Dropdown selectbox for choosing backend model
    model_choice = st.sidebar.selectbox(
        "Choose an LLM backend",
        ["ollama: llama3:3b", "ollama: mistral:7b", "openai: gpt-4o-mini", "anthropic: claude-3-sonnet"]
    )
    # Build the question-answering chain with the chosen model
    qa = build_qa_chain(model_choice)
except Exception as e:
    error_msg = str(e)  # Store error message if backend fails

st.sidebar.markdown("---")  # Separator line in sidebar
st.sidebar.header("📊 Data Filters")  # Filters section title

# --- Year filter ---
# Get min and max year from first_air_date column
min_year = int(df["first_air_date"].dropna().str[:4].min())
max_year = int(df["first_air_date"].dropna().str[:4].max())
# Slider to choose range of years
year_range = st.sidebar.slider("Year Range", min_year, max_year, (min_year, max_year))

# --- Genre filter ---
# Build a sorted unique list of all genres
all_genres = sorted(set(df["genres"].dropna().str.split(",").explode()))
# Multiselect for user to pick genres
selected_genres = st.sidebar.multiselect("Select Genres", all_genres)

# --- Platform filter ---
# Predefined binary platform columns
platforms = ["Netflix", "Hulu", "Prime Video", "Disney+"]
# Multiselect for user to pick platforms
selected_platforms = st.sidebar.multiselect("Select Platforms", platforms)


# ---------------------------
# Apply Filters
# ---------------------------

# Work on a copy of the dataset
df_filtered = df.copy()

# Filter by year range
df_filtered = df_filtered[
    df_filtered["first_air_date"].str[:4].fillna("0").astype(int).between(year_range[0], year_range[1])
]

# Filter by genres (keep rows where at least one selected genre matches)
if selected_genres:
    df_filtered = df_filtered[
        df_filtered["genres"].dropna().apply(lambda g: any(genre in g for genre in selected_genres))
    ]

# Filter by platforms (keep rows where any selected platform column is True/1)
if selected_platforms:
    mask = df_filtered[selected_platforms].any(axis=1)
    df_filtered = df_filtered[mask]


# ---------------------------
# Tabs: Conditional
# ---------------------------

if qa:  # If LLM backend is available
    tab1, tab2 = st.tabs(["🔎 Ask Questions", "📊 Data Analysis"])

    # --- Tab 1: RAG Q&A ---
    with tab1:
        st.header("Ask Questions about TV Shows")
        query = st.text_input("Enter your question:")  # Input field for user query
        if query:
            result = qa.invoke(query)  # Run the RAG chain with the query
            st.write("### Answer")
            st.write(result["result"])  # Show model’s answer
            st.write("### Sources")
            # Show sources (documents) used for the answer (truncated to 400 chars each)
            for doc in result["source_documents"]:
                st.write(doc.page_content[:400] + "...")
                st.write("---")

    # --- Tab 2: Seaborn Analysis ---
    with tab2:
        st.header("TV Show Dataset Analysis")

else:
    # If no LLM backend is available, show only data analysis
    st.warning("⚠️ No LLM backend available. Showing only data analysis.")
    st.header("📊 TV Show Dataset Analysis")


# ---------------------------
# Shared Seaborn Analysis (uses df_filtered)
# ---------------------------

# --- Release Year Trend ---
st.subheader("📈 Release Year Trend (Top 20)")
fig, ax = plt.subplots(figsize=(12, 6))
sns.countplot(
    data=df_filtered,
    x="first_air_date",
    order=df_filtered["first_air_date"].value_counts().index[:20],  # top 20 years
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
st.pyplot(fig)

# --- Genre Distribution ---
st.subheader("🎭 Top 15 Genres")
fig, ax = plt.subplots(figsize=(10, 6))
df_filtered["genres"].dropna().str.split(",").explode().value_counts()[:15].plot(
    kind="bar", color="skyblue", ax=ax
)
st.pyplot(fig)

# --- Rating Distribution ---
st.subheader("⭐ Rating Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df_filtered["vote_average"], bins=20, kde=True, color="purple", ax=ax)
st.pyplot(fig)

# --- Country Distribution ---
st.subheader("🌍 Top 10 Producing Countries")
fig, ax = plt.subplots(figsize=(12, 6))
df_filtered["origin_country"].dropna().str.split(",").explode().value_counts()[:10].plot(
    kind="bar", color="orange", ax=ax
)
st.pyplot(fig)
