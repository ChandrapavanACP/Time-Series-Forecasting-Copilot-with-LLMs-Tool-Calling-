# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Make a AI Forecasting Agent that includes a preprocessor and react forecasting agent.

# LIBRARIES

from langchain_openai import ChatOpenAI

import os
import yaml

import pandas as pd

from IPython.display import Image, display, Markdown

from forecast_team.agents.forecast_agent import make_forecast_agent

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# INPUTS:

MODEL   = "gpt-4.1-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# DATA

df_sample = pd.read_csv("data/walmart_sales_monthly.csv")

# * TESTING

app = make_forecast_agent(
    model=llm,
    sample_nrow=100,
)   

app

# Display ReAct Agent Mermaid Graph

Image(app.get_graph(xray=1).draw_mermaid_png())


# * TESTING

result = app.invoke({
    "user_question": "Aggregate sales by month for each food item. Make a forecast for the next 12 months.",
    "data_sample_sql": df_sample.head(20).to_dict(),
    "data_sql": df_sample.to_dict(),
})

result

list(result.keys())

Markdown(result['formatted_user_question_forecast_only'])

pd.DataFrame(result['data_forecast'])

Markdown(result['forecast_summary'])
