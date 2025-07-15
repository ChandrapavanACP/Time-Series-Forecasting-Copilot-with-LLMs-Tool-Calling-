# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage

from langgraph.graph import START, END, StateGraph

from typing import Sequence, TypedDict

import pandas as pd

from forecast_team.agents.sql_agent import make_sql_agent
from forecast_team.agents.forecast_agent import make_forecast_agent
from forecast_team.agents.plot_forecast_agent import make_plot_forecast_agent

from forecast_team.utils import get_last_human_message

import os
import yaml

os.environ["POLARS_SKIP_CPU_CHECK"] = "1"
os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']


# KEY INPUTS 
PATH_DB = "sqlite:///data/walmart_sales.db"
MODEL   = "gpt-4.1-mini"


# TEAMS:

def make_forecast_team(
    path=PATH_DB,  
    model=MODEL, 
    sample_nrow=1000,
    checkpointer=None,
):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        model = ChatOpenAI(model = model)
        
    llm = model 
    
    # SUB-AGENTS
    sql_agent = make_sql_agent(
        path=path,
        model=llm,
        sample_nrow=sample_nrow,
    )

    forecast_agent = make_forecast_agent(
        model=llm,
        sample_nrow=sample_nrow,
    )

    plot_forecast_agent = make_plot_forecast_agent(
        model=llm,
    )
    
   
    # LANGGRAPH
    class ForecastTeamState(TypedDict):
        messages: Sequence[BaseMessage] # NEW - list that holds the chat history
        response: Sequence[BaseMessage] # NEW - list that holds the agent's response
        user_question: str
        # New Agent - Routing Agent
        forecast_agent_required: str
        # SQL Agent
        formatted_user_question_sql_only: str
        sql_query : str
        data_sample_sql: dict
        data_sql: dict
        # Forecast Agent
        formatted_user_question_forecast_only: str
        data_sample_forecast: dict
        data_forecast: dict
        forecast_summary: str
        # Chart Agent
        formatted_user_question_chart_only: str
        plot_forecast_json: dict
        plot_forecast_summary: str
        # New Summary Agent
        summary: str


    def prepare_user_question(state):
        
        # Get the user question and chat history
        messages = state.get("messages")
        
        last_human_question = get_last_human_message(messages)
        if last_human_question:
            last_human_question = last_human_question.content
            
        return {"user_question": last_human_question}
        

    def route_to_forecast_agent(state):
        print("---ROUTER---")
        
        # * Preprocessor to determine routing
        route_preprocessor_prompt = PromptTemplate(
            template="""
                Your job is to determine the Agent Routing to a Forecast Agent. 
                
                Please look for if the user has asked for a forecast or not.
                
                If they have, then return 'Yes' for "forecast_agent_required".
                If they have not, then return 'No' for "forecast_agent_required".
                
                Example of forecast being required:
                Group the products by bike.description road vs mountain and call this group id - you'll need to check the description for Road or Mountain which will be present in the text. Aggregate the sales by month in each group in id. Forecast the next 24 months for each group by id. When plotting, make sure to use the id as the unique id.
                
                Example of no forecast being required:
                Group the products by bike.description road vs mountain and call this group id - you'll need to check the description for Road or Mountain which will be present in the text. Aggregate the sales by month in each group in id.             
            
                Return JSON with 'Yes' or 'No' for "forecast_agent_required".
                
                USER_QUESTION: {question}
            """,
            input_variables=['question']
        )
        
        route_preprocessor = route_preprocessor_prompt | llm | JsonOutputParser()
        
        response = route_preprocessor.invoke({"question": state['user_question']})
        
        print(f"    * Router Response: {str(response)}")
        
        return {'forecast_agent_required': response['forecast_agent_required']}

    # SUMMARY AGENT
    def summarize_results(state):
        
        print("--- SUMMARIZE RESULTS ---")
        
        summarizer_prompt = PromptTemplate(
            template="""
            You are an expert in summarizing the analysis results of a Forecasting Expert. Your goal is to help the business understand the analysis in basic terms that business people can easily understand. Be consice in your explanation of the results. When possible please include summary tables to convey information instead of bullet points. Do not use markdown headers in your response.

            The Forecasting Expert has knowledge of the company's sales data and forecasting models. Can write SQL, produce forecasts in table and charts. Has access to the sales SQL database that includes SQL tables containing information on sales, products, and customer behavior.

            You are given the results of a the Forecasting Expert's analysis. Your job is to summarize the results in a way that is easy to understand for business people. 
            
            ANALYSIS RESULTS FOR SUMMARIZATION: 
            {results}
            """,
            input_variables=["results"]
        )

        summarizer = summarizer_prompt | llm | StrOutputParser()
        
        summary_payload = {
            "user_question": state.get("user_question"),
            "sql_query": state.get("sql_query"),
            "forecast_agent_required": state.get("forecast_agent_required"),
            "data_sample_sql": pd.DataFrame(state.get("data_sample_sql", [])).head(100).to_dict(orient='records'),
            "forecast_summary": state.get("forecast_summary"),
            "plot_forecast_summary": state.get("plot_forecast_summary"),
        }
        
        summary_results = summarizer.invoke({
            "results": summary_payload
        })
        
        return {
            "summary": summary_results,
            "response": [AIMessage(content=summary_results, name="Forecast_Team_Agent")]
        }

    # * WORKFLOW DAG

    workflow = StateGraph(ForecastTeamState)

    workflow.add_node("prepare_user_question", prepare_user_question)
    workflow.add_node("sql_agent", sql_agent)
    workflow.add_node("forecast_agent", forecast_agent)
    workflow.add_node("plot_forecast_agent", plot_forecast_agent)
    workflow.add_node("route_to_forecast_agent", route_to_forecast_agent)
    workflow.add_node("summarize_results", summarize_results)

    workflow.add_edge(START, "prepare_user_question")
    workflow.add_edge("prepare_user_question", "sql_agent")
    workflow.add_edge("sql_agent", "route_to_forecast_agent")

    workflow.add_conditional_edges(
        "route_to_forecast_agent",
        lambda state: state['forecast_agent_required'],
        {"Yes": "forecast_agent", "No": "__end__"}
    )

    workflow.add_edge("forecast_agent", "plot_forecast_agent")
    workflow.add_edge("plot_forecast_agent", "summarize_results")
    workflow.add_edge("summarize_results", END)

    # Checkpointer for Short Term Memory
    app = workflow.compile(checkpointer=checkpointer)

    return app
