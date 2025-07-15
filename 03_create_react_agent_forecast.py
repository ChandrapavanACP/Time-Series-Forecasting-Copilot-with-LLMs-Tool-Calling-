# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI TIME SERIES FORECASTING AGENT
# ***

# GOAL: Demonstrate the steps in making a ReAct Agent that can forecast time series data using LangChain and LangGraph.

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

from pprint import pprint
from IPython.display import Image, display, Markdown

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

# INPUTS

MODEL   = "gpt-4.1-mini"

# AI SETUP

os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

llm = ChatOpenAI(model = MODEL)

# DATA

data_dict = pd.read_csv("data/walmart_sales_monthly.csv").to_dict()

pd.DataFrame(data_dict)


# * STEP 1: MAKE A FORECAST TOOL

@tool(response_format="content_and_artifact")
def generate_forecast(
    data: Annotated[dict, InjectedState("data_sql")], 
    id_col: str, 
    date_col: str, 
    value_col: str, 
    forecast_horizon: int,
) -> Tuple[str, Dict]:
    """
    Tool: generate_forecast
    
    Description:
    Forecasts time series data using XGBoost and returns a DataFrame with predictions and confidence intervals.
    
    Parameters:
    - data (dict): dict containing the time series data.
    - id_col (str): Column name for item IDs.
    - date_col (str): Column name for dates.
    - value_col (str): Column name for the values to forecast.
    - forecast_horizon (int): Number of periods to forecast into the future.

    Returns:
    - Tuple[str, Dict]: A tuple containing a content string explaining the output and an artifact dictionary with the forecasted data.
    """
    
    print("    * Tool: generate_forecast")
    
    # Wrap imports inside function to make scope local:
    import numpy as np
    import pandas as pd
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from pytimetk import future_frame

    # Convert input dictionary to DataFrame
    df = pd.DataFrame(data)

    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Get unique item IDs
    unique_items = df[id_col].unique()

    # Wrap functions to enforce local scope for exec() command:
    def extend_single_timeseries_frame(data, id_col, date_col, value_col, id_value, h=60):        

        # Filter for one time series
        df_single = data[data[id_col] == id_value].copy()
        df_single[date_col] = pd.to_datetime(df_single[date_col])

        # Actual data with ACTUAL label
        df_actual = df_single.assign(key="ACTUAL")

        # Generate future dates using pytimetk
        df_future = (
            future_frame(df_single, date_column=date_col, length_out=forecast_horizon, bind_data=True)
            .assign(**{value_col: np.nan, "key": "FUTURE"})
            .tail(forecast_horizon)
        )

        # Combine actual and future
        combined_df = pd.concat([df_actual, df_future], ignore_index=True)

        return combined_df
    
    def make_timeseries_features(data, date_col):
        """Adds engineered time series features from the datetime column."""
        data['_index.num'] = pd.to_datetime(data[date_col]).astype(np.int64) // 10**9
        data['_year'] = pd.to_datetime(data[date_col]).dt.year
        data['_month'] = pd.to_datetime(data[date_col]).dt.month
        data['_mday'] = pd.to_datetime(data[date_col]).dt.day
        data['_wday'] = pd.to_datetime(data[date_col]).dt.dayofweek
        return data


    def forecast_single_timeseries(data, id_col, date_col, value_col, id_value, h=60, **kwargs):
        "Forecasts a single timeseries using XGBoost."
        
        single_timeseries_extended_df = extend_single_timeseries_frame(data, id_col, date_col, value_col, id_value, h)
        
        single_timeseries_feat_df = make_timeseries_features(single_timeseries_extended_df, date_col=date_col)

        future_df = single_timeseries_feat_df.query("key == 'FUTURE'")
        actual_df = single_timeseries_feat_df.query("key == 'ACTUAL'")

        df = actual_df.drop([id_col, 'key'], axis=1)
        X = df.drop([value_col, date_col], axis=1)
        y = df[value_col]

        # Calculate Conformal Interval
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=h, random_state=123, shuffle=False)

        model = XGBRegressor(**kwargs)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        std_dev = np.std(abs(y_test - y_pred))

        # Refit to full dataset
        X_train = actual_df.drop([id_col, 'key', value_col, date_col], axis=1)
        y_train = actual_df[value_col]

        model.fit(X_train, y_train)

        # Predict on future data
        future_df[value_col] = model.predict(future_df.drop([id_col, 'key', value_col, date_col], axis=1))

        future_df['conf_lo'] = future_df[value_col] - 1.96 * std_dev
        future_df['conf_hi'] = future_df[value_col] + 1.96 * std_dev

        combined_df = pd.concat([actual_df, future_df], axis=0)
        return combined_df

    # Initialize list for forecasted DataFrames
    forecast_dfs = []

    # Loop over each unique item and make forecasts
    for item in unique_items:
        forecast_df = forecast_single_timeseries(
            data=df,
            id_col=id_col,
            date_col=date_col,
            value_col=value_col,
            id_value=item,
            h=forecast_horizon
        )
        forecast_dfs.append(forecast_df)

    # Combine all forecasts into a single DataFrame
    combined_forecast_df = pd.concat(forecast_dfs, ignore_index=True)

    # Remove the engineered features
    combined_forecast_df = combined_forecast_df.drop(columns=[col for col in combined_forecast_df.columns if col.startswith('_')])
    
    # * New: Create content, artifact for Tool response
    
    # Create content that explains the output data, columns, and how to interpret it
    content = f"""
The dataset is `{len(combined_forecast_df)} rows` long and contains forecasts for the next `{forecast_horizon}` months for each item. The confidence intervals provide a range in which the actual values are expected to fall, with a 95% confidence level.
    
The forecasted data contains the following columns:

    - `{id_col}`: Unique identifier for each item.
    - `{date_col}`: The date of the forecast.
    - `{value_col}`: The forecasted total sales for each item.
    - `conf_lo`: The lower bound of the confidence interval for the forecasted value.
    - `conf_hi`: The upper bound of the confidence interval for the forecasted value.
    """

    artifact = {
        "forecast_data": combined_forecast_df.to_dict()
    }
    
    # Return Format: Content and Artifact as Tuple
    return content, artifact


# * TESTING THE TOOL

# Exploring the LangChain StructuredTool

# ?generate_forecast

generate_forecast

generate_forecast.name

Markdown(generate_forecast.description)

generate_forecast.func

# Test the tool's function 

forecast_content_and_artifact = generate_forecast.func(
    data = data_dict,
    id_col = "item_id",
    date_col = "year_month",
    value_col = "total_value",
    forecast_horizon = 12
)

Markdown(forecast_content_and_artifact[0])

pd.DataFrame(forecast_content_and_artifact[1]['forecast_data'])


# * STEP 2: MAKE A FORECAST REACT AGENT

# Custom Agent State for Forecasting Agent
class ForecastState(AgentState):
    """
    Represents the state of our forecasting agent.
    """
    data_sql: dict

forecast_react_agent = create_react_agent(
    model=llm,
    tools=[generate_forecast],
    state_schema=ForecastState,
    name="Forecasting_React_Agent",
)

forecast_react_agent

# Examine the Agent's State Schema
pprint(forecast_react_agent.get_output_jsonschema()['properties'])

# Test the react agent with a simple question

result = forecast_react_agent.invoke(
    {
        "messages": [HumanMessage(content="Make a forecast for the next 12 months of total sales for each item. ID Column: item_id, Date Column: year_month, Value Column: total_value.")], 
        "data_sql": data_dict,
    }
)

result.keys()

result['messages']

# Tool Messages

# Get the last tool message
last_tool_message = result['messages'][-2]

# Name
last_tool_message.name

# Content
Markdown(last_tool_message.content)

# Artifacts

artifacts = last_tool_message.artifact

artifacts.keys()

pd.DataFrame(artifacts['forecast_data'])


# AI Message

ai_message = result['messages'][-1]

Markdown(ai_message.content)





