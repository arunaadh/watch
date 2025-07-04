import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent import AgentExecutor
from langchain_core.exceptions import OutputParserException
import httpx
import re

# --- Configure HTTPX to bypass SSL verification (for dev only) ---
# os.environ["HTTPX_OPTIONAL_CLIENT_TLS_VERIFY"] = "0"
# http_client = httpx.Client(verify=False)

# --- Load environment variables ---
load_dotenv()
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "").strip()
DB_NAME = os.getenv("DB_NAME", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, OPENAI_API_KEY]):
    raise ValueError("❌ Missing environment variables in .env")

# --- DB Setup ---
connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=disable"
engine = create_engine(connection_string)
db = SQLDatabase(engine=engine)

# --- OpenAI Chat LLM Setup ---
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    openai_api_key=OPENAI_API_KEY,
    http_client=http_client
)

# --- Agent Setup ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    handle_parsing_errors=True
)

if not isinstance(agent_executor, AgentExecutor):
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent_executor.agent,
        tools=agent_executor.tools,
        verbose=True,
        handle_parsing_errors=True
    )

# --- Streamlit UI Setup ---
st.set_page_config(page_title="⌚ Watch Data Analyst", layout="wide")
st.title("⌚ AI Analytics")

# --- Column Descriptions Sidebar ---
column_descriptions = {
    "id": "A unique identifier for each watch listing.",
    "listing_code": "A unique code or identifier for the listing, typically provided by the platform (e.g., Chrono24).",
    "brand": "The brand name of the watch (e.g., Rolex, Omega, Patek Philippe).",
    "model": "The specific model name or number of the watch (e.g., Submariner, Speedmaster).",
    "reference_number": "The manufacturer's reference number used to uniquely identify the watch configuration.",
    "avg_price_usd": "The average price of the watch in USD, calculated from multiple listings (if applicable).",
    "case_size_mm": "The size of the watch case, typically in millimeters (e.g., '40mm', '36 mm').",
    "material": "The primary material used in the watch case (e.g., Stainless Steel, Gold, Titanium).",
    "movement": "The type of movement inside the watch (e.g., Automatic, Quartz, Manual).",
    "year_of_production": "The year the watch was manufactured or released.",
    "condition": "The condition of the watch (e.g., New, Like New, Pre-owned).",
    "gender": "The intended gender for the watch (e.g., Men's, Women's, Unisex).",
    "availability": "The stock status of the watch (e.g., Available, Sold, Reserved).",
    "image_url": "A URL pointing to an image of the watch, usually hosted on the listing website.",
    "scrape_timestamp": "The timestamp indicating when the listing was scraped and stored in the database.",
    "url": "The URL of the original watch listing page on the source platform."
}

with engine.connect() as conn:
    cols = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'watch_listings'
        ORDER BY ordinal_position
    """)).fetchall()

    st.sidebar.markdown("### 📋 `Watch data` Description")
    for col_name, in cols:
        st.sidebar.write(f"🔹 `{col_name}`: {column_descriptions.get(col_name, 'No description available.')}")

# --- Main UI ---
user_query = st.chat_input("Ask a question about the watch data available in your database:")
if user_query:
    with st.spinner("Thinking..."):
        try:
            agent_response_obj = None
            try:
                agent_response_obj = agent_executor.invoke({"input": user_query})
                result = agent_response_obj.get('output', '').strip()
            except OutputParserException:
                st.warning("⚠️ The agent couldn't format the final answer. Try rephrasing.")
                st.stop()

            st.success("Answer:")
            st.markdown(result)

            # SQL Extraction Prompt
            sql_prompt = (
                f"Generate only the PostgreSQL query for the following question: {user_query}. "
                "If using GROUP BY, apply aggregate functions like AVG or COUNT on numeric columns. "
                "Return it in a markdown block like ```sql\n...``` with no explanation."
            )
            sql_response = agent_executor.invoke({"input": sql_prompt})
            raw_sql = sql_response.get("output", "").strip()

            sql_match = re.search(r"```sql\s+(.*?)\s+```", raw_sql, re.DOTALL)
            if sql_match:
                sql_query = sql_match.group(1)
            else:
                st.info("ℹ️ Could not extract SQL query from the agent output.")
                st.stop()

            df = pd.read_sql(text(sql_query), engine)

            if df.empty:
                st.info("ℹ️ No results found.")
            elif df.shape[1] == 2:
                df.columns = ['x', 'y']
                fig, ax = plt.subplots()
                if pd.api.types.is_numeric_dtype(df['x']) or pd.api.types.is_datetime64_any_dtype(df['x']):
                    df.plot(kind='line', x='x', y='y', ax=ax, marker='o', legend=False)
                else:
                    df.plot(kind='bar', x='x', y='y', ax=ax, legend=False)
                st.pyplot(fig)
            elif df.shape[1] == 1:
                if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                    df.hist()
                    st.pyplot()
                else:
                    st.dataframe(df)
            else:
                st.dataframe(df)

        except Exception as e:
            st.error("❌ Something went wrong.")
            st.code(str(e), language="text")
            traceback.print_exc()
