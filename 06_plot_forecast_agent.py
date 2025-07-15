# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Make a AI Plot Forecasting Agent that includes a preprocessor and react plotting agent.

# LIBRARIES

from langchain_openai import ChatOpenAI

import os
import yaml

import pandas as pd
import plotly.io as pio

from pprint import pprint
from IPython.display import Image, display, Markdown

from forecast_team.agents.plot_forecast_agent import make_plot_forecast_agent

# INPUTS:

MODEL   = "gpt-4.1-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# DATA
df_sample = pd.read_csv("data/walmart_sales_monthly_forecast.csv")

# * TESTING

app = make_plot_forecast_agent(llm)

app

# Display ReAct Agent Mermaid Graph

Image(app.get_graph(xray=1).draw_mermaid_png())

result = app.invoke({
    "user_question": "Aggregate sales by month for each food item. Make a forecast for the next 12 months.",
    "data_sample_forecast": df_sample.head(20).to_dict(),
    "data_forecast": df_sample.to_dict(),
})

result

list(result.keys())

Markdown(result['formatted_user_question_chart_only'])

# Display the plot
pio.from_json(result['plot_forecast_json'])

