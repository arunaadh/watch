import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import traceback
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import URL
import re # Import regex module for parsing

import httpx
from openai import OpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_openai import ChatOpenAI

# --- Configure httpx client to skip SSL verification for LLM connection ---
# This must be set BEFORE any httpx client is initialized (which happens internally).
# WARNING: Bypassing SSL verification is NOT recommended for production.
# This is for local development on organizational laptops with SSL inspection.
try:
    # Ensure httpx is installed: pip install httpx
    # For persistent SSL issues on corporate networks, setting this env var can help.
    # It tells httpx (used by OpenAI's client) to not verify SSL certificates.
    os.environ["HTTPX_OPTIONAL_CLIENT_TLS_VERIFY"] = "0" # "0" or "false" to disable verification
    #st.warning("⚠️ SSL verification for HTTPX (used by OpenAI API) is bypassed via environment variable. Use only for local development.")

    # --- IMPORTANT: Check and set proxy environment variables if you are behind a corporate proxy ---
    # If your organization uses an HTTP/HTTPS proxy, ensure these environment variables are set
    # in your system or before running the Streamlit app. Example (in terminal):
    # export HTTP_PROXY="http://your.proxy.com:8080"
    # export HTTPS_PROXY="http://your.proxy.com:8080"
    # Or for Windows:
    # set HTTP_PROXY=http://your.proxy.com:8080
    # set HTTPS_PROXY=http://your.proxy.com:8080
    # If you have specific proxy credentials:
    # export HTTPS_PROXY="http://user:password@your.proxy.com:8080"
    #if os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY"):
    #    st.info(f"Detected HTTP_PROXY: {os.getenv('HTTP_PROXY')} and HTTPS_PROXY: {os.getenv('HTTPS_PROXY')}. Ensure these are correct for your network.")
    #else:
    #    st.info("No HTTP_PROXY or HTTPS_PROXY environment variables detected. If you are behind a corporate proxy, you might need to set them.")

    http_client = httpx.Client(verify=False) # Create the client to inject into ChatOpenAI
except Exception as e:
    st.error(f"❌ Failed to configure httpx client for SSL bypass: {e}")
    st.stop() # Stop the app if this critical step fails

# --- Load environment variables ---
load_dotenv()

DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = os.getenv("DB_PORT", "").strip()
DB_NAME = os.getenv("DB_NAME", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, OPENAI_API_KEY]):
    raise ValueError("❌ Missing environment variables in .env. Ensure DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, and OPENAI_API_KEY are set.")

# --- SQLAlchemy DB Engine (PostgreSQL connection) ---
connection_string = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    "?sslmode=disable" # Keep this if your PostgreSQL connection also had SSL issues on your laptop
)
engine = create_engine(connection_string)
db = SQLDatabase(engine=engine)

# --- LangChain OpenAI LLM setup with injected http_client ---
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo", # Or "gpt-4" if you have access and prefer it
    openai_api_key=OPENAI_API_KEY,
    http_client=http_client # Inject the custom httpx client
)

# --- LangChain SQL Agent ---
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# --- Streamlit UI ---
st.set_page_config(page_title="⌚ Watch Data Analyst", layout="wide")
st.title("⌚ AI Analytics")

# Show schema in sidebar
with engine.connect() as conn:
    try:
        # Define the descriptive mapping for columns
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

        cols = conn.execute(text("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'watch_listings'
            ORDER BY ordinal_position
        """)).fetchall()

        st.sidebar.markdown("### 🔍 Available Data Fields")
        st.sidebar.markdown("---") # Add a separator for neatness
        for col_name_tuple in cols:
            col_name = col_name_tuple[0]
            description = column_descriptions.get(col_name, "No description available.")
            st.sidebar.markdown(f"**{col_name.replace('_', ' ').title()}**: {description}")
            st.sidebar.markdown("") # Add a small space between descriptions for readability

    except Exception as schema_err:
        st.sidebar.error("❌ Could not load schema.")
        print("Schema error:", schema_err)

# Initialize session state for query management
if 'user_query_submitted' not in st.session_state:
    st.session_state.user_query_submitted = False
    st.session_state.last_query = ""
    # Initialize the value that controls the text area content
    st.session_state.current_query_input = ""

# --- Main logic for displaying results ---
if st.session_state.user_query_submitted:
    user_query = st.session_state.last_query
    query = f"Using table `public.watch_listings`, answer: {user_query}"
    with st.spinner("Thinking..."):
        try:
            # Get LLM Answer
            result = agent_executor.run(query)
            st.success("Analysis Result:")
            st.markdown(result)

            # --- MODIFIED: Get SQL Query and Parse it with more robust regex ---
            sql_prompt = (
                f"Generate only the PostgreSQL query for the following question, "
                f"do not include any other text or explanation. "
                f"Wrap the SQL query in a markdown code block like this: ```sql\nSELECT ...;```\n\n"
                f"Question: {user_query}"
            )
            raw_sql_output = agent_executor.invoke({"input": sql_prompt})['output']

            # Print raw output for debugging
            print(f"--- Raw LLM SQL Output for Debugging ---\n{raw_sql_output}\n---------------------------------------")

            # Updated regex:
            # - `.*?` at the beginning to match any leading characters non-greedily.
            # - ```sql\s*` matches the opening tag, allowing any whitespace after 'sql'.
            # - `([\s\S]*?)` captures the SQL content (including newlines) non-greedily.
            # - ``` ` matches the closing tag.
            # - `re.DOTALL` ensures `.` matches newlines.
            match = re.search(r".*?```sql\s*([\s\S]*?)```", raw_sql_output, re.DOTALL)
            if match:
                sql = match.group(1).strip()
                # st.code(sql, language="sql") # This line remains commented out as per previous request
            else:
                # If no SQL block is found, display the raw output and raise an error
                st.warning("⚠️ Could not extract a valid SQL query from the LLM's response. Raw output:")
                st.code(raw_sql_output, language="text")
                raise ValueError("LLM did not return a valid SQL query in the expected format.")

            # Execute SQL
            df = pd.read_sql(text(sql), engine)

            # --- Only show chart/data section if DataFrame is not empty ---
            if not df.empty:
                st.subheader("📊 Visualization or Data")
                if df.shape[1] == 2:
                    df.columns = ['x', 'y']
                    fig, ax = plt.subplots()
                    # Determine if it should be a line chart or bar chart
                    if pd.api.types.is_numeric_dtype(df['x']) or pd.api.types.is_datetime64_any_dtype(df['x']):
                        df.plot(kind='line', x='x', y='y', ax=ax, legend=False, marker='o')
                        st.pyplot(fig)
                    else:
                        # Otherwise, default to bar chart
                        df.plot(kind='bar', x='x', y='y', ax=ax, legend=False)
                        st.pyplot(fig)
                elif df.shape[1] == 1:
                    # Check if the single column is numeric before attempting hist
                    if pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                        df.hist()
                        st.pyplot()
                    else:
                        st.warning("⚠️ Cannot plot histogram: the column contains non-numeric data.")
                        st.dataframe(df)
                else:
                    st.dataframe(df)
            else:
                st.info("ℹ️ No results found for your query. The database returned an empty set.")

        except Exception as err:
            st.error("❌ Failed to answer or visualize. Error details:")
            st.code(str(err), language="text")
            print("❌ FULL ERROR:")
            traceback.print_exc()

# --- Main input at the bottom with a submit button ---
with st.form(key='query_form'):
    # --- MODIFIED: Added placeholder and controlled value for clearing ---
    user_query_input = st.text_area(
        "Ask any question about the watch models available in your database:",
        height=100,
        placeholder="e.g., 'What is the average price of Rolex watches by year of production?' or 'Show me the brands with the most listings.'", # Predefined query example
        value=st.session_state.current_query_input, # Control the value to clear it
        key="query_input_box_form"
    )
    submit_button = st.form_submit_button(label='Get Insights') # Changed button text

# Process the query only when the submit button is pressed
if submit_button and user_query_input:
    st.session_state.user_query_submitted = True
    st.session_state.last_query = user_query_input
    st.session_state.current_query_input = "" # Clear the input box after submission
    st.rerun() # Rerun the app to process the query and display results
elif submit_button and not user_query_input:
    st.warning("Please enter a query before submitting.")

# Reset submission state if the input box is cleared manually after submission
if st.session_state.get('user_query_submitted') and not st.session_state.get('query_input_box_form'):
    st.session_state.user_query_submitted = False
    st.session_state.last_query = ""
