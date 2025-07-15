# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langgraph.graph import END, StateGraph

import os

from typing import TypedDict

import pandas as pd
import sqlalchemy as sql

from forecast_team.utils import SQLOutputParser


# KEY INPUTS 
PATH_DB  = "sqlite:///data/walmart_sales.db"
MODEL    = "gpt-4o-mini"

# * SQL AGENT
def make_sql_agent(
    path = PATH_DB,  
    model = MODEL, 
    sample_nrow = 1000, 
):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model    
    
    # DATABASE SETUP
    db = SQLDatabase.from_uri(path)

    # * AGENTS

    # * SQL Preprocessor Agent

    sql_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent .
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
        
        
        Return JSON with 'formatted_user_question_sql_only'.
        
        USER_QUESTION: {question}
        """,
        input_variables=["question"]
    )

    sql_preprocessor = sql_preprocessor_prompt | llm | JsonOutputParser()

    # * SQL Agent            

    prompt_sql_generator = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run. Then, look at the results of the query and return the answer to the input question.

        IMPORTANT SQL RULES:

        * If the user asks for a forecast, time series, or aggregation, if the data has an identifier column (e.g. item_id, product_id, department), include it in the SELECT clause. If there are multiple identifier columns, include all of them and create a composite identifier.

        * If the user asks for a forecast, time series, or aggregation over time, and no ID column is available, then:

            - Add a synthetic identifier using something like:
            'TOTAL' AS "series_id"

            - Create a synthetic "series_id" column that is constant for all rows in the result set.
            
            - This is required to support downstream forecasting tools that expect an ID column.

        * If the user is asking a metadata question (like "what tables exist?" or "what is in the sales table?"), or no time series aggregation is being performed, then DO NOT add a synthetic "series_id" column.

        * Never use `SELECT *`. Always select only the necessary columns. Wrap each column name in double quotes (") to use them as delimited identifiers.

        * Do not use a LIMIT clause unless the user asks for it. Avoid using the {top_k} value unless required.

        * Use `date('now')` for current date comparisons.

        * Return the SQL inside ```sql``` tags.

        * Only return one query. Do not explain the SQL.

        * Examples:
        
            1. test question: "Aggregate total sales by day."
            
            Expected Output:
            ```sql
            SELECT 
                'TOTAL' AS "series_id", 
                "date", 
                SUM("sales") AS "total_value"
            FROM "sales"
            GROUP BY "date"
            ```
            
            2. test question: "What are the first 10 rows in the daily_demand table?"

            Expected Output:
            ```sql
            SELECT * FROM "daily_demand"
            LIMIT 10
            ```
            3. test question: "What tables exist in the database?"
            
            Expected Output:
            ```sql
            SELECT name FROM sqlite_master WHERE type='table';
            ```
            
        9. Only use the following tables and columns:
        {table_info}

        User Question: {input}
        
        """
    )

    sql_generator = (
        create_sql_query_chain(
            llm = llm,
            db = db,
            k = int(1e7),
            prompt = prompt_sql_generator
        ) 
        | SQLOutputParser() 
    )


    # * LANGGRAPH

    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        user_question: str
        formatted_user_question_sql_only: str
        sql_query : str
        data_sample_sql: dict
        data_sql: dict
        
    def preprocess_sql(state):
        print("---SQL GENERATOR---")
        print("    * PREPROCESS FOR SQL GENERATOR")
        question = state.get("user_question")
        
        # SQL Prep
        response = sql_preprocessor.invoke({"question": question})
        
        return {
            "formatted_user_question_sql_only": response['formatted_user_question_sql_only'],
        }
        

    def generate_sql(state):
        print("    * GENERATE SQL")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        # Generate Data Frame
        sql_engine = sql.create_engine(path)
        conn = sql_engine.connect()
        sql_query_2 = sql_query.rstrip("'")
        df = pd.read_sql(sql_query_2, conn)
        
        print(df.head(10).to_string())
        
        conn.close()
        
        return {
            "sql_query": sql_query, 
            "data_sample_sql": df.head(sample_nrow).to_dict(),
            "data_sql": df.to_dict()
        }


    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)
    
    workflow.add_node("preprocess_sql", preprocess_sql)
    workflow.add_node("generate_sql", generate_sql)
    
    workflow.set_entry_point("preprocess_sql")
    workflow.add_edge("preprocess_sql", "generate_sql")
    workflow.add_edge("generate_sql", END)

    app = workflow.compile()
    
    return app