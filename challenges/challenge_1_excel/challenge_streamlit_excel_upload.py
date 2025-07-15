# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING TEAM
# ***

# CHALLENGE: ALLOW THE USER TO UPLOAD A CSV OR EXCEL FILE INSTEAD OF CONNECTING TO A DATABASE

# DIFFICULTY: INTERMEDIATE

# SPECIFIC ACTIONS:
#  1. Allow the user to upload a CSV or Excel file
#  2. Create a temporary SQLite file-based database
#  3. Show uploaded data preview
#  4. Allow the user to select an OpenAI model

# QUESTIONS:
# - If using the walmart_sales.xlsx file

# What tables exist in the database?

# What are the first 10 rows in the daily_demand table?

# Aggregate sales by month for each food item. Make a forecast for the next 12 months.

# Collect the data for FOODS_3_090. Aggregate sales by day. Make a forecast for the next 365 days.

# Aggregate sales by day. Make a forecast for the next 365 days.

# Aggregate sales by day for each food item. Make a forecast for the next 365 days. Do not include a legend in the forecast.

# IMPORTANT HINTS:
# - Use the `sqlite3` library to create a temporary SQLite database.
# - Use the `pandas` library to read the uploaded CSV or Excel file and convert it to a SQLite database.
# - Use sys and pathlib to allow streamlit to locate your forecast_team module.

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
