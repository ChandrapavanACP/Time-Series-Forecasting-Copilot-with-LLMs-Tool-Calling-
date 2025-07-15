# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Make a SQL AI Agent to collect and prepare data for the forecasting agent


# LIBRARIES

from langchain_openai import ChatOpenAI

import os
import yaml

import pandas as pd

from IPython.display import Image, display, Markdown

from forecast_team.agents.sql_agent import make_sql_agent

# INPUTS:

PATH_DB = "sqlite:///data/walmart_sales.db"
MODEL   = "gpt-4.1-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# * TESTING

app = make_sql_agent(
    path = PATH_DB,
    model = llm, 
    sample_nrow = 100,
)

app

display(Image(data=app.get_graph().draw_png()))

# Question 1

result = app.invoke({"user_question": "What tables exist in the database?"})
result

list(result.keys())

result['sql_query']

Markdown("``` sql\n" + result['sql_query'] + "\n```")

pd.DataFrame(result['data_sql'])

# Question 2

result = app.invoke({"user_question": "What are the first 10 rows in the daily_demand table?"})
result

result['sql_query']

Markdown("``` sql\n" + result['sql_query'] + "\n```")

pd.DataFrame(result['data_sql'])

# Question 3

result = app.invoke({"user_question": "Aggregate sales by month for each food item. Make a forecast for the next 12 months."})
result.keys()

print(result['formatted_user_question_sql_only'])

result['sql_query']

Markdown("``` sql\n" + result['sql_query'] + "\n```")

pd.DataFrame(result['data_sql'])

