# AI DATA SCIENCE TEAM - EDA TOOL CALLING AGENT DEMO
# ==========================================================
# This demo shows how to use the AI Data Science Team's EDA Tools Agent to perform exploratory data analysis (EDA) using multiple tools like Sweetviz, Dtale, correlation funnel, and more


# Resources:
# 1. Tool Examples: https://github.com/business-science/ai-data-science-team/blob/master/ai_data_science_team/tools/eda.py
# 2. Example: https://github.com/business-science/ai-data-science-team/blob/master/examples/ds_agents/eda_tools_agent.ipynb

# LIBRARIES

# AI Libraries
from langchain_openai import ChatOpenAI

# Import necessary libraries
import pandas as pd
import os
import yaml

# Agent
from ai_data_science_team.ds_agents import EDAToolsAgent

# SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model="gpt-4o-mini")
llm

# DATASET
df = pd.read_csv("data/churn_data.csv")
df

# AGENT

# Make a EDA agent
exploratory_agent = EDAToolsAgent(
    llm, 
    invoke_react_agent_kwargs={"recursion_limit": 10},
)
exploratory_agent

# TOOLS

exploratory_agent.invoke_agent("What tools do you have access to? Return a table.")

exploratory_agent.get_ai_message(markdown=True)

exploratory_agent.response

# USING A TOOL

exploratory_agent.invoke_agent(
    "Generate a sweetviz report for the Churn dataset. Use Churn as the main feature.",
    data_raw=df,
)

# EXPLORING RESPONSES FROM A TOOL-CALLING AGENT (AKA REACT AGENT)

exploratory_agent.response

exploratory_agent.get_internal_messages()

result = exploratory_agent.response

result['eda_artifacts']
