# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import END, StateGraph

# Tool Calling
from langchain_core.messages import HumanMessage
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.prebuilt import create_react_agent

from typing import TypedDict

import pandas as pd

from forecast_team.tools import generate_forecast
from forecast_team.utils import get_last_tool_message

# KEY INPUTS 
MODEL    = "gpt-4o-mini"



# * FORECAST AGENT
def make_forecast_agent(model = MODEL, sample_nrow = 1000):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model   
    
    # * Preprocessor Agent
    
    forecast_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in interpreting directions for a Time Series Forecasting Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a Forecast Generator Agent.
        2. Determine the names of the id column, date column, and a value column based on the sample data provided. 
        3. If making a forecast, determine the forecast horizon (number of periods to forecast).
        
        Return JSON with 'formatted_user_question_forecast_only'.
        
        Example:
        User Question: "Aggregate sales by month for each food item. Make a forecast for the next 12 months."
        Output: 
        {{
            "formatted_user_question_forecast_only": "Make a forecast for the next 12 months. id_column = item_id, date_column= month, value_column= sales,forecast_horizon= 12"
        }}
        
        USER_QUESTION: {question}
        
        DATA SAMPLE SQL: {data_sample_sql}
        """,
        input_variables=["question", "data_sample_sql"]
    )

    forecast_preprocessor = forecast_preprocessor_prompt | llm | JsonOutputParser()
            
    # * Forecast ReAct Agent
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
    
    # * LANGGRAPH

    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        data_sample_sql: dict
        data_sql: dict
        formatted_user_question_forecast_only: str
        data_sample_forecast: dict
        data_forecast: dict
        forecast_summary: str
        
    def preprocess_forecast(state):
        print("---FORECASTER---")
        print("    * PREPROCESS FOR FORECAST GENERATOR")
        
        question = state.get("user_question")
        
        data_sample_sql = state.get("data_sample_sql")
        
        # Forecast prep
        response = forecast_preprocessor.invoke({
            "question": question, 
            "data_sample_sql": data_sample_sql
        })
        
        # print(response['formatted_user_question_forecast_only'])
        
        return {
            "formatted_user_question_forecast_only": response['formatted_user_question_forecast_only'],
        }

    def generate_forecast_data(state):
        print("    * GENERATE FORECAST DATA")
        
        question = state.get("formatted_user_question_forecast_only")
        data_dict = state.get("data_sql")
        
        result = forecast_react_agent.invoke(
            {
                "messages": [HumanMessage(content=question)], 
                "data_sql": data_dict,
            }
        )
        
        # print(result['messages'])
        
        # Get the last tool message
        last_tool_message = get_last_tool_message(result['messages'], target_name="generate_forecast")
        
        if not last_tool_message:
            return {"forecast_summary": "No forecast data generated. Please check the input data and question."}

        # Artifacts
        artifacts = last_tool_message.artifact
        
        forecast_df = pd.DataFrame(artifacts['forecast_data'])
        
        return {
            "data_sample_forecast": forecast_df.head(sample_nrow).to_dict(),
            "data_forecast": forecast_df.to_dict(),
            "forecast_summary": last_tool_message.content,            
        }
        

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_forecast", preprocess_forecast)
    workflow.add_node("generate_forecast_data", generate_forecast_data)
  
    workflow.set_entry_point("preprocess_forecast")
    workflow.add_edge("preprocess_forecast", "generate_forecast_data")
    workflow.add_edge("generate_forecast_data", END)

    app = workflow.compile()
    
    return app
