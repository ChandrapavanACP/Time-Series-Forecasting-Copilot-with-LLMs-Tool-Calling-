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

from forecast_team.tools import plot_forecast
from forecast_team.utils import get_last_tool_message

# KEY INPUTS
MODEL = "gpt-4o-mini"


def make_plot_forecast_agent(
    model = MODEL,
):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model
        
    # * Preprocessor Agent
    chart_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in interpreting directions for a Charting Agent. Your job is to:
        
        1. Request a plot for the forecast
        2. Determine the names of the id column, date column, and a value column based on the sample data provided.
        
        Return JSON with 'formatted_user_question_chart_only'.
        
        Example:
        User Question: "Make a sales forecast by month for each food item."
        Output: 
        {{
            "formatted_user_question_chart_only": "Plot sales forecast by month. id_column = item_id, date_column= month, value_column= sales"
        }}
        
        USER_QUESTION: {question}

        DATA SAMPLE FORECAST: {data_sample_forecast}
        """,
        input_variables=["question", "data_sample_forecast"]
    )
    
    chart_preprocessor = chart_preprocessor_prompt | llm | JsonOutputParser()
    
    # * Plot Forecast ReAct Agent
    
    class PlotForecastState(AgentState):
        data_forecast: dict

    plot_forecast_react_agent = create_react_agent(
        model=llm,
        tools=[plot_forecast],
        state_schema=PlotForecastState,
        name="Plot_Forecast_React_Agent",
    )

    # * LANGGRAPH

    class GraphState(TypedDict):
        """
        Represents the state of our Graph.
        """
        user_question: str
        formatted_user_question_chart_only: str
        data_sample_forecast: dict
        data_forecast: dict
        plot_forecast_json: dict
        plot_forecast_summary: str
        
        
    def preprocess_plot(state):
        print("---CHART AGENT---")
        print("    * PREPROCESS FOR CHART GENERATOR")
        
        question = state.get("user_question")

        data_sample_forecast = state.get("data_sample_forecast")

        # Forecast prep
        response = chart_preprocessor.invoke({
            "question": question,
            "data_sample_forecast": data_sample_forecast
        })
        
        # print(response['formatted_user_question_chart_only'])
        
        return {
            "formatted_user_question_chart_only": response['formatted_user_question_chart_only'],
        }
    

    def generate_plot_data(state):
        print("    * GENERATE PLOT DATA")
        
        # Get the formatted user question for charting
        formatted_user_question_chart_only = state.get("formatted_user_question_chart_only")
        
        # Get the data forecast
        data_forecast = state.get("data_forecast")
        
        # Invoke the plot forecast react agent
        result = plot_forecast_react_agent.invoke({
            "messages": [HumanMessage(content=formatted_user_question_chart_only)],
            "data_forecast": data_forecast,
        })
        
        # print(result['messages'])
        
        # Get the last tool message
        last_tool_message = get_last_tool_message(result['messages'], target_name="plot_forecast")
        
        if not last_tool_message:
            return {"plot_forecast_summary": "No plot generated."}
        
        # Artifacts
        artifacts = last_tool_message.artifact
        
        if 'plotly_json' not in artifacts:
            return {"plot_forecast_summary": "No plotly JSON found in artifacts."}
        
        # Return the plotly JSON and summary content
        content = last_tool_message.content
        
        return {
            "plot_forecast_json": artifacts['plotly_json'],
            "plot_forecast_summary": content,
        }
        
    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_plot", preprocess_plot)
    workflow.add_node("generate_plot_data", generate_plot_data)

    workflow.set_entry_point("preprocess_plot")
    workflow.add_edge("preprocess_plot", "generate_plot_data")
    workflow.add_edge("generate_plot_data", END)

    app = workflow.compile()
    
    return app
