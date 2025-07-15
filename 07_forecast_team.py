# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Make a simple logic-based router that handles routing tasks to SQL and Forecasting

# ADVANTAGES:
# 1. Logic-Based Execution is easy to set up (Clinic #2)
# 2. Typically Faster that Supervision (Clinic #3)
# 3. Not as flexible. Supervision allows much more complex workflows.

# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

import os
import yaml
import pandas as pd
import plotly.io as pio

from IPython.display import Image, display, Markdown

from forecast_team.teams import make_forecast_team

# Add Short Term Memory
from langgraph.checkpoint.memory import MemorySaver

# KEY INPUTS 
PATH_DB = "sqlite:///data/walmart_sales.db"
MODEL   = "gpt-4.1-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# * CREATE THE FORECAST TEAM
# SEE `forecast_team/teams.py` for the full team implementation.

# * TESTING

forecast_team = make_forecast_team(
    path=PATH_DB,
    model=MODEL,
    sample_nrow=1000
)

forecast_team

display(Image(forecast_team.get_graph(xray=2).draw_mermaid_png()))


# QUESTIONS 

# Question 1
result_1 = forecast_team.invoke({"messages": [HumanMessage(content="What tables exist in the database?")]})

list(result_1.keys())

result_1['sql_query']
pd.DataFrame(result_1['data_sql'])

result_1['forecast_agent_required']


# Question 2
result_2 = forecast_team.invoke({"messages": [HumanMessage(content="What are the first 10 rows in the daily_demand table?")]})

result_2['sql_query']
pd.DataFrame(result_2['data_sql'])


# Question 3
result_3 = forecast_team.invoke({"messages": [HumanMessage(content="Aggregate sales by month for each food item. Make a forecast for the next 12 months.")]})

list(result_3.keys())

pio.from_json(result_3['plot_forecast_json'])

pd.DataFrame(result_3['data_forecast'])

Markdown(result_3['summary'])


# Question 4
result_4 = forecast_team.invoke({"messages": [HumanMessage(content="Collect the data for FOODS_3_090. Aggregate sales by day. Make a forecast for the next 365 days.")]})

list(result_4.keys())

pio.from_json(result_4['plot_forecast_json'])

pd.DataFrame(result_4['data_sql'])

pd.DataFrame(result_4['data_forecast'])

Markdown(result_4['summary'])

result_4['formatted_user_question_forecast_only']


# Question 5

result_5 = forecast_team.invoke({"messages": [HumanMessage(content="Aggregate sales by day. Make a forecast for the next 365 days.")]})

list(result_5.keys())

pio.from_json(result_5['plot_forecast_json'])


# Question 6
result_6 = forecast_team.invoke({"messages": [HumanMessage(content="Aggregate sales by day for each food item. Make a forecast for the next 365 days. Do not include a legend in the forecast.")]})

list(result_6.keys())

pio.from_json(result_6['plot_forecast_json'])
