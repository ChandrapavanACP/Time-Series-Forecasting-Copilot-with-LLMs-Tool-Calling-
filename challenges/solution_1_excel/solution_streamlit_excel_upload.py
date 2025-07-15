# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Integrate the Agential Workflow into a streamlit app

# Command Line:
#   streamlit run path/to/app.py

# LIBRARIES

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import pandas as pd
import plotly.io as pio
import streamlit as st

# * NEW: Add Project Root:
import sys
from pathlib import Path

# * IMPORTANT - Add project root directory to sys.path
project_root = Path(__file__).resolve().parents[2]  # Adjust number based on depth from root
sys.path.append(str(project_root))

from forecast_team.teams import make_forecast_team

import os
import yaml

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# * New: Solution - Streamlit Excel Upload App
import sqlite3
import io
import tempfile

# * STREAMLIT APP SETUP ----

CHAT_LLM_OPTIONS = [
    # "gpt-4.1-nano", # POOR RESULTS WITH COMPLEX TASKS LIKE FORECASTING
    "gpt-4.1-mini", 
    "gpt-4.1", 
    "gpt-4o-mini", 
    "gpt-4o"
]

# * Page Setup

st.set_page_config(page_title="Your AI Time Series Forecasting Agent")
st.title("Your AI Time Series Forecasting Agent")

# Top-level description
with st.expander("I'm a handy forecasting AI agent that connects to multiple SQLite databases. You can ask me questions about the SQL database, perform aggregations, and make forecasts. I will report the results. (see example questions)"):
    # Replace nested expanders with tabs
    tab1, tab2 = st.tabs(["Walmart Database Questions", "Bike Shop Database Questions"])
    
    with tab1:
        st.markdown(
            """            
            **Walmart Database Questions:**

            1. What tables exist in the database?

            2. Show me the `daily_demand` table.
            
            3. What are the total sales for each food item by month? Make sure to show the date column.

            4. Aggregate sales by month for each food item. Make a forecast for the next 12 months.

            5. Collect the data for `FOODS_3_090`. Aggregate sales by day. Make a forecast for the next 365 days.

            6. Aggregate sales by day. Make a forecast for the next 365 days.

            7. Aggregate sales by day for each food item. Make a forecast for the next 365 days. Do not include a legend in the forecast.
            """
        )
    
    with tab2:
        st.markdown(
            """
            **Bike Shop Database Questions:**

            1. What tables exist in the database?

            2. Show me the first 10 rows of the `orderlines` table with all columns shown.

            3. Show me the first 10 rows of the `bikes` table with all columns shown.

            4. Show me the first 10 rows of the `bikeshops` table with all columns shown.

            5. Aggregate the sales data by multiplying the bike unit price × the quantity and sum by day. Forecast the transactions by day for the next 365 days.  

            6. Aggregate the sales data by multiplying the bike unit price × the quantity and sum by month. Forecast the transactions by month for the next 24 months. 

            7. Group the products by `bike.description` road vs mountain and call this group `id`. Aggregate the sales by month in each group in `id`. Forecast the next 24 months for each group by `id`. When plotting, make sure to use the `id` as the unique id. 
            """
        )


# * New: Solution - Streamlit Excel Upload App (Replace database with excel upload and temp database)

# Allow user to upload CSV or Excel file
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    # Read the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)

    # Create a temporary SQLite file-based database
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as temp_db:
        PATH_DB = temp_db.name
        conn = sqlite3.connect(PATH_DB)
        df.to_sql("uploaded_data", conn, index=False, if_exists="replace")
        PATH_DB="sqlite:///"+PATH_DB
        conn.close()  # Close the connection to ensure data is written to disk
    
    # Show uploaded data preview
    st.write("Preview of uploaded data:")
    st.dataframe(df)
else:
    st.write("Please upload a CSV or Excel file to proceed.")

# * Database Option

# db_option = st.sidebar.selectbox(
#     "Select a Database",
#     ["Walmart Sales Database", "Bike Shop Database"]
# )

# db_options = {
#     "Walmart Sales Database": "sqlite:///data/walmart_sales.db",
#     "Bike Shop Database": "sqlite:///data/bikeshop_database.sqlite",
# }

# PATH_DB = db_options.get(db_option)

# * model selection

# Sidebar for model selection
model_option = st.sidebar.selectbox(
    "Choose OpenAI model",
    CHAT_LLM_OPTIONS,
    index=0
)

OPENAI_LLM = ChatOpenAI(
    model = model_option,
)

llm = OPENAI_LLM

# MAKE THE FORECAST TEAM

# forecast_team = make_forecast_team(
#     path = PATH_DB, 
#     model = llm,
#     sample_nrow=100,
# )

# * STREAMLIT 

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

# Initialize plot storage in session state
if "plots" not in st.session_state:
    st.session_state.plots = []

# Initialize dataframe storage in session state
if "dataframes" not in st.session_state:
    st.session_state.dataframes = []

# Initialize details storage in session state
if "details" not in st.session_state:
    st.session_state.details = []

# Function to display chat messages including Plotly charts and dataframes
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "PLOT_INDEX:" in msg.content:
                plot_index = int(msg.content.split("PLOT_INDEX:")[1])
                st.plotly_chart(st.session_state.plots[plot_index])
            elif "DETAILS_INDEX:" in msg.content:
                detail_index = int(msg.content.split("DETAILS_INDEX:")[1])  # get the index
                detail = st.session_state.details[detail_index]             # select the correct detail

                with st.expander("Forecast details:", expanded=True):
                    st.markdown("### Forecast Details")
                    tab1, tab2 = st.tabs(["AI Reasoning", "SQL Query"])

                    with tab1:
                        st.markdown("### AI Reasoning")
                        st.markdown(detail["ai_reasoning"])

                    with tab2:
                        st.markdown("### SQL Query")
                        st.markdown(detail["sql_query"])

            elif "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)

# Render current messages from StreamlitChatMessageHistory
display_chat_history()

if question := st.chat_input("Enter your question here:", key="query_input"):
    with st.spinner("Thinking..."):
        
        # * NEW - Move forecast team inside the question loop
        forecast_team = make_forecast_team(
            path = PATH_DB, 
            model = llm,
            sample_nrow=100,
        )
        
        st.chat_message("human").write(question)
        msgs.add_user_message(question)
        
        # Run the app       
        error_occured = False
        try: 
            print(PATH_DB)
            result = forecast_team.invoke(
                input={
                    "messages": [HumanMessage(content=question)]
                }, 
            )
        except Exception as e:
            error_occured = True
            print(e)
        
        # Generate the Results
        if not error_occured:
            
            sql_query = result.get("sql_query")
            data_sql = result.get("data_sql")
            data_forecast = result.get("data_forecast")
            plot_forecast_json = result.get("plot_forecast_json")
            
            
            if plot_forecast_json:
                
                response_1 = """
                ### Forecast Results:
                Here is a forecast plot and downloadable forecast data. 
                """
                
                # Store the plot and keep its index
                response_plot = pio.from_json(plot_forecast_json)
                plot_index = len(st.session_state.plots)
                st.session_state.plots.append(response_plot)
                
                # Store the forecast details
                detail_index = len(st.session_state.details)
                response_text = {
                    "ai_reasoning": result.get("summary", "No AI reasoning provided."),
                    "sql_query": f"```sql\n{sql_query}\n```",
                }
                st.session_state.details.append(response_text)
                
                # Store the forecast df and keep its index
                response_df = pd.DataFrame(data_forecast)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)
                
                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"PLOT_INDEX:{plot_index}")
                msgs.add_ai_message(f"DETAILS_INDEX:{detail_index}")
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                
                # Write the results
                st.chat_message("ai").write(response_1)
                st.plotly_chart(response_plot)
                with st.expander("Forecast details:"):
                    if "details" in st.session_state and st.session_state.details:
                        # Grab ONLY the last (most recent) detail
                        current_detail_index = len(st.session_state.details) - 1
                        detail = st.session_state.details[current_detail_index]

                        st.markdown("### Forecast Details")
                        tab1, tab2 = st.tabs(["AI Reasoning", "SQL Query"])

                        with tab1:
                            st.markdown("### AI Reasoning")
                            st.markdown(detail["ai_reasoning"])

                        with tab2:
                            st.markdown("### SQL Query")
                            st.markdown(detail["sql_query"])
                
                st.dataframe(response_df)
            
            elif sql_query:
                
                # Store the SQL
                response_1 = f"### SQL Results:\n\nSQL Query:\n\n```sql\n{sql_query}\n```\n\nResult:"
                
                # Store the forecast df and keep its index
                response_df = pd.DataFrame(data_sql)
                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)

                # Store response
                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")
                
                # Write Results
                st.chat_message("ai").write(response_1)
                st.dataframe(response_df)
                
            
                
        else:
            # An error occurred
            response_text = f"I'm sorry. I am having difficulty answering that question. You can try providing more details and I'll do my best to provide an answer."
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)


