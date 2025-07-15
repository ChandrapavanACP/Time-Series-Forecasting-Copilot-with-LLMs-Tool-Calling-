# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Make a AI Charting Agent to convert the forecast into a chart

# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Tool Calling
from langchain.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent

import os
import yaml

from typing import Annotated, Dict, Tuple

import pandas as pd
import plotly.io as pio

from pprint import pprint
from IPython.display import Image, display, Markdown

# INPUTS

MODEL   = "gpt-4o-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# DATA

data_dict = pd.read_csv("data/walmart_sales_monthly_forecast.csv").to_dict()

pd.DataFrame(data_dict)


# * STEP 1: CREATE THE PLOT FORECAST TOOL

@tool(response_format="content_and_artifact")
def plot_forecast(
    data: Annotated[dict, InjectedState("data_forecast")],  
    id_col: str, 
    date_col: str, 
    value_col: str
) -> Tuple[str, Dict]:
    """
    Tool: plot_forecast
    
    Description:
    Plots the forecasted time series data with confidence intervals using Plotly.
    
    Parameters:
    - data (dict): dict containing the time series data with the forecast confidence intervals.
    - id_col (str): Column name for item IDs.
    - date_col (str): Column name for dates.
    - value_col (str): Column name for the values to plot.
    
    Returns:
    - Tuple[str, Dict]: A tuple containing a content string describing the plot and a dictionary with the Plotly figure serialized as JSON.
    """
    
    print("    * Tool: plot_forecast")
    
    import pandas as pd
    import plotly.graph_objs as go
    import plotly.io as pio

    # Ensure the DataFrame is created from the provided data
    df = pd.DataFrame(data)

    # Define the required column names

    conf_lo_col = 'conf_lo'
    conf_hi_col = 'conf_hi'
    
    unique_ids = df[id_col].unique()

    # Create an initial figure
    fig = go.Figure()

    # Add traces for each item_id
    for i, item_id in enumerate(unique_ids):
        group = df[df[id_col] == item_id]

        # Add the main forecast line for each item
        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[value_col], 
            mode='lines', 
            showlegend=False,
            name='Forecast - ' + str(item_id),
            visible=True if i == 0 else False
        ))

        # Add the upper and lower bounds as filled area for each item
        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[conf_hi_col], 
            mode='lines', 
            name='Upper Bound - ' + str(item_id),
            line=dict(width=0),
            showlegend=False,
            visible=True if i == 0 else False
        ))

        fig.add_trace(go.Scatter(
            x=group[date_col], 
            y=group[conf_lo_col], 
            mode='lines', 
            name='Lower Bound - ' + str(item_id),
            fill='tonexty', 
            fillcolor='rgba(173,216,230,0.3)',  # Light blue fill
            line=dict(width=0),
            showlegend=False,
            visible=True if i == 0 else False
        ))

    # Create a dropdown to toggle between the forecasts
    dropdown_buttons = [
        dict(
            label=str(item_id),
            method="update",
            args=[dict(visible=[i // 3 == idx for i in range(len(unique_ids) * 3)]),
                   dict(title="Forecasts with Confidence Intervals - " + str(item_id))]
        )
        for idx, item_id in enumerate(unique_ids)
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                active=0,
                buttons=dropdown_buttons,
                x=0,
                xanchor="left",
                y=1.15,
                yanchor="top",
                showactive=True,
            )
        ],
        hovermode="x unified",
        title="Forecast - " + str(unique_ids[0]),
        xaxis_title=date_col,
        yaxis_title=value_col,
    )
    
    # * New: create content and artifact for the Tool to return
    
    content = f"""
    A plot has been created showing the forecasted values for each item over time, along with confidence intervals. The plot allows you to toggle between different items using the dropdown menu.
    """
    
    artifact = {
        "plotly_json": pio.to_json(fig),
    }

    return content, artifact

# * TESTING THE TOOL

plot_content_and_artifact = plot_forecast.func(
    data = data_dict,
    id_col = "item_id",
    date_col = "year_month",
    value_col = "total_value"
)

plot_content_and_artifact[0]  # Content
plot_content_and_artifact[1]['plotly_json']  # Artifact

pio.from_json(plot_content_and_artifact[1]['plotly_json'])


# * STEP 2: CREATE THE REACT AGENT

class PlotForecastState(AgentState):
    data_forecast: dict

plot_forecast_react_agent = create_react_agent(
    model=llm,
    tools=[plot_forecast],
    state_schema=PlotForecastState,
    name="Plot_Forecast_React_Agent",
)

plot_forecast_react_agent


# Test the react agent with a simple question

result = plot_forecast_react_agent.invoke(
    {
        "messages": [HumanMessage(content="Plot a forecast. ID Column: item_id, Date Column: year_month, Value Column: total_value.")], 
        "data_forecast": data_dict,
    }
)

result.keys()

result['messages']


# Get the last tool message
last_tool_message = result['messages'][-2]

# Name
last_tool_message.name

# Content
Markdown(last_tool_message.content)

# Artifacts

artifacts = last_tool_message.artifact

artifacts.keys()

pio.from_json(artifacts['plotly_json'])

