# *** BUSINESS INTELLIGENCE EXPERT (CLINIC #2) ***

# Key Modifications:
# 1. Routing Preprocessor Agent: Now gets Messages History "chat_history"
# 2. New Summarizer Step: Added to summarize the analysis results, summary gets returned to the Supervisor Agent

from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, AIMessage

from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI

import pandas as pd
import sqlalchemy as sql
import plotly.io as pio

from pprint import pprint

from typing import Sequence, TypedDict

from marketing_analytics_team.agents.utils import get_last_human_message
from marketing_analytics_team.agents.utils import SQLOutputParser, PythonOutputParser

# KEY INPUTS

def make_business_intelligence_agent(model, db_path):    
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
    
    PATH_DB = db_path
    
    llm.temperature = 0
    
    # * Routing Preprocessor Agent
    #   * NEW: CONTEXT "chat_history" added to the prompt
    
    routing_preprocessor_prompt = PromptTemplate(
        template="""
        You are an expert in routing decisions for a SQL database agent, a Charting Visualization Agent, and a Pandas Table Agent. Your job is to:
        
        1. Determine what the correct format for a Users Question should be for use with a SQL translator agent 
        2. Determine whether or not a chart should be generated or a table should be returned based on the users question.
        
        Use the following criteria on how to route the the initial user question:
        
        From the incoming user question, remove any details about the format of the final response as either a Chart or Table and return only the important part of the incoming user question that is relevant for the SQL generator agent. This will be the 'formatted_user_question_sql_only'. If 'None' is found, return the original user question.
        
        Next, determine if the user would like a data visualization ('chart') or a 'table' returned with the results of the SQL query. If unknown, not specified or 'None' is found, then select 'table'.  
        
        Return JSON with 'formatted_user_question_sql_only' and 'routing_preprocessor_decision'.
        
        INITIAL_USER_QUESTION: {initial_question}
        
        CONTEXT: {chat_history}
        """,
        input_variables=["initial_question", "chat_history"]
    )

    routing_preprocessor = routing_preprocessor_prompt | llm | JsonOutputParser()

    routing_preprocessor


    # * SQL Agent

    db = SQLDatabase.from_uri(PATH_DB)

    # SQL Output Parser

    prompt_sqlite = PromptTemplate(
        input_variables=['input', 'table_info', 'top_k'],
        template="""
        You are a SQLite expert. Given an input question, first create a syntactically correct SQLite query to run, then look at the results of the query and return the answer to the input question.
        
        Do not use a LIMIT clause with {top_k} unless a user specifies a limit to be returned.
        
        Return SQL in ```sql ``` format.
        
        Only return a single query if possible.
        
        Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
        
        Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
        
        Pay attention to use date(\'now\') function to get the current date, if the question involves "today".
            
        Only use the following tables:
        {table_info}
        
        Question: {input}'
        """
    )

    sql_generator = (
        create_sql_query_chain(
            llm = llm,
            db = db,
            k = int(1e7),
            prompt = prompt_sqlite
        ) 
        | SQLOutputParser() # NEW SQLCodeExtactor
    )


    # * Chart Instructor Agent

    prompt_chart_instructions = PromptTemplate(
        template="""
        You are a supervisor that is an expert in providing instructions to a chart generator agent for plotting. 
        
        You will take a question that a user has and the data that was generated to answer the question, and create instructions to create a chart from the data that will be passed to a chart generator agent.
        
        USER QUESTION: {question}
        
        DATA: {data}
        
        Formulate "chart generator instructions" by informing the chart generator of what type of plotly plot to use (e.g. bar, line, scatter, etc) to best represent the data. 
        
        Come up with an informative title from the user's question and data provided. Also provide X and Y axis titles.
        
        Instruct the chart generator to use the following theme colors, sizes, etc:
        
        - Use this color for bars and lines:
            'blue': '#3381ff',
        - Base Font Size: 8.8 (Used for x and y axes tickfont, any annotations, hovertips)
        - Title Font Size: 13.2
        - Line Size: 0.65 (specify these within the xaxis and yaxis dictionaries)
        - Add smoothers or trendlines to scatter plots unless not desired by the user
        - Do not use color_discrete_map (this will result in an error)
        - Hover tip size: 8.8
        
        Return your instructions in the following format:
        CHART GENERATOR INSTRUCTIONS: FILL IN THE INSTRUCTIONS HERE
        
        """,
        input_variables=['question', 'data']
    )

    chart_instructor = prompt_chart_instructions | llm | StrOutputParser()


    # * Chart Generator Agent

    prompt_chart_generator = PromptTemplate(
        template = """
        You are an expert in creating data visualizations and plots using the plotly python library. You must use plotly or plotly.express to produce plots. Your job is to produce python code to generate visualizations.
        
        # IMPORTANT NOTES:
        
        1. Return a single function named, "plot_chart" that ingests a parameter containing "data", and outputs the plotly fig.
        2. Return Python code in ```python ``` format.
        3. Important: Keep the scope of the plot_chart() function local (imports and helper functions inside the main function). This makes it easier to use this function with exec()
        
        CHART INSTRUCTIONS: 
        {chart_instructions}
        
        INPUT DATA: 
        {data}
        
        EXAMPLE FUNCTION CODE TO RETURN (USE THIS FORMAT):
        
        ```python
        def plot_chart(data):
        
            # Import Libraries inside function
            import pandas as pd
            import plotly.express as px
            
            # Create Plot
            fig = px.bar(data, x='Category', y='Value')
            
            return fig
        ```
        
        Important Notes on creating the chart code:
        - Do not use color_discrete_map. This is an invalid property.
        - If bar plot, do not add barnorm='percent' unless user asks for it
        - If bar plot, do not add a trendline. Plotly bar charts do not natively support the trendline.  
        - For line plots, the line width should be updated on traces (example: # Update traces
    fig.update_traces(line=dict(color='#3381ff', width=0.65)))
        - For Bar plots, the default line width is acceptable
        
        """,
        input_variables=["chart_instructions", "data"]
    )

    chart_generator = prompt_chart_generator | llm | PythonOutputParser()
    
    # * NEW: Summarizer
    
    summarizer_prompt = PromptTemplate(
        template="""
        You are an expert in summarizing the analysis results of a Customer Transactions Expert. Your goal is to help the business understand the analysis in basic terms that business people can easily understand. Be consice in your explanation of the results. When possible please include summary tables to convey information instead of bullet points. Do not use markdown headers in your response.
        
        The Customer Transactions Expert as knowledge of the company's customer transactions database. Has analytics and business intelligence skills. Can write SQL, produce data in table and charts. Has access to the customer SQL database that includes SQL tables containing information on customers, lead scores (how likely they are to buy), transactions, courses purchased, and types of products.
        
        You are given the results of a the Customer Transaction Expert's analysis that contain:
        
        - user_question: The initial user question that was asked to the Customer Transactions Expert
        - chat_history: The previous chat history provided for additional context on the user's question
        - formatted_user_question_sql_only: A processed version of the user question provided to the SQL expert
        - sql_query: The sql query that the SQL expert created by accessing the Customer Analytics Database
        - data: The results of the sql query when run on the database
        - routing_processor_decision: either 'table' or 'chart'. If table, a chart is not returned. If chart, plotly code is created. 
        
        If a 'chart' was determined, the application will attempt to produce a chart. Sometimes errors occur, which is denoted by: 'chart_plotly_error'
        
        If a chart was successful, Python code and JSON will be produced, which are contained in 'chart_plotly_code' and 'chart_plotly_json' respectively. 
         
        ANALYSIS RESULTS FOR SUMMARIZATION: {results}
        """,
        input_variables=["results"]
    )

    summarizer = summarizer_prompt | llm | StrOutputParser()

    # * LANGGRAPH
    class GraphState(TypedDict):
        """
        Represents the state of our graph.
        """
        messages: Sequence[BaseMessage] # NEW - list that holds the chat history
        response: Sequence[BaseMessage] # NEW - list that holds the agent's response
        user_question: str
        formatted_user_question_sql_only: str
        sql_query : str
        data: dict
        routing_preprocessor_decision: str
        chart_generator_instructions: str
        chart_plotly_code: str
        chart_plotly_json: dict
        chart_plotly_error: bool
        summary: str # NEW - summary of the analysis results
        
    def preprocess_routing(state):
        print("---BUSINESS INTELLIGENCE EXPERT---")
        print("    * PREPROCESSOR AND ROUTING")
        
        # Get the user question and chat history
        messages = state.get("messages")
        
        last_human_question = get_last_human_message(messages)
        if last_human_question:
            last_human_question = last_human_question.content
        
        # Chart Routing and SQL Prep
        response = routing_preprocessor.invoke({"initial_question": last_human_question, "chat_history": messages})
        
        # print("formatted user question sql only:")
        # pprint(response['formatted_user_question_sql_only'])
        
        formatted_user_question_sql_only = response['formatted_user_question_sql_only']
        
        routing_preprocessor_decision = response['routing_preprocessor_decision']
        
        return {
            "user_question": last_human_question,
            "formatted_user_question_sql_only": formatted_user_question_sql_only,
            "routing_preprocessor_decision": routing_preprocessor_decision,
        }
        
    def generate_sql(state):
        print("    * GENERATE SQL")
        question = state.get("formatted_user_question_sql_only")
        
        # Handle case when formatted_user_question_sql_only is None:
        if question is None:
            question = state.get("user_question")
        
        # Generate SQL
        sql_query = sql_generator.invoke({"question": question})
        
        return {"sql_query": sql_query}

    def convert_dataframe(state):
        print("    * CONVERT DATA FRAME")

        sql_query = state.get("sql_query")
        
        # pprint(state)
        
        # pprint(sql_query)
        
        # Generate Data Frame
        sql_engine = sql.create_engine(PATH_DB)
        conn = sql_engine.connect()
        sql_query_2 = sql_query.rstrip("'")
        df = pd.read_sql(sql_query_2, conn)
        conn.close()
        
        return {"data": df.to_dict(orient="records")}

    def decide_chart_or_table(state):
        print("    * DECIDE CHART OR TABLE")
        return "chart" if state.get('routing_preprocessor_decision') == "chart" else "table"

    def instruct_chart_generator(state):
        print("    * INSTRUCT CHART GENERATOR")
        
        # Get the user question and data
        question = state.get("user_question")
        data = state.get("data")
        
        # if data is large, sample
        df = pd.DataFrame(data)
        if df.shape[0] > 1000:
            data = df.sample(1000).to_dict()
        
        chart_generator_instructions = chart_instructor.invoke({"question": question, "data": data})
        
        return {"chart_generator_instructions": chart_generator_instructions}

    def generate_chart(state):
        print("    * GENERATE CHART")
        
        # Get the chart generator instructions and data
        chart_instructions = state.get("chart_generator_instructions")
        data = state.get("data")
        
        # if data is large, sample
        df = pd.DataFrame(data)
        if df.shape[0] > 1000:
            data = df.sample(1000).to_dict()
        
        # Generate Chart Python Code
        response = chart_generator.invoke({"chart_instructions": chart_instructions, "data": data})
        
        # Execute the chart code and get the JSON
        chart_plotly_error = False
        fig_json = None
        if "error" in response[:40].lower():
            chart_plotly_error = True
        else:
            try:
                # Create dictionaries to hold the local and global variables
                local_vars = {}
                global_vars = {}
                
                exec(response, global_vars, local_vars)
                
                plot_chart = local_vars.get("plot_chart")

                fig = plot_chart(df)
                
                fig_json = pio.to_json(fig)
            except:
                chart_plotly_error = True
            
        return {
            "chart_plotly_code": response, 
            "chart_plotly_json": fig_json, 
            "chart_plotly_error": chart_plotly_error,
        }
    
    # * NEW: Summarizer Node
    def summarize_results(state):
        print("    * SUMMARIZE RESULTS")
        
        # Build a minimal payload for summarization to reduce token usage
        # Error code: 429 - {'error': {'message': 'Request too large for gpt-4.1-mini-long-context in organization org-ljnmniNYtkh3c5B8LFIFv5ZF on tokens per min (TPM): Limit 400000, Requested 1647293. The input or output tokens must be reduced in order to run successfully. Visit https://platform.openai.com/account/rate-limits to learn more.', 'type': 'tokens', 'param': None, 'code': 'rate_limit_exceeded'}}
        summary_payload = {
            "user_question": state.get("user_question"),
            "sql_query": state.get("sql_query"),
            "routing_processor_decision": state.get("routing_preprocessor_decision"),
            # include up to first 10 rows for context
            "data_preview": state.get("data", [])[:100],
            "chart_plotly_code": state.get("chart_plotly_code"),
        }
        
        result = summarizer.invoke({"results": summary_payload})
        
        return {
            "summary": result,
            "response": [AIMessage(content=result, name='Business_Intelligence_Expert')],
        }
        
    def state_printer(state):
        print("    * STATE PRINTER")
        print(f"User Question: {state['user_question']}")
        print(f"Formatted Question (SQL): {state['formatted_user_question_sql_only']}")
        print(f"SQL Query: \n{state['sql_query']}\n")
        print(f"Data: \n{pd.DataFrame(state['data'])}\n")
        print(f"Chart or Table: {state['routing_preprocessor_decision']}")
        
        if state['routing_preprocessor_decision'] == "chart":
            print(f"Chart Code: \n{pprint(state['chart_plotly_code'])}")
            print(f"Chart Error: {state['chart_plotly_error']}")
        

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("preprocess_routing", preprocess_routing)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("convert_dataframe", convert_dataframe)
    workflow.add_node("instruct_chart_generator", instruct_chart_generator)
    workflow.add_node("generate_chart", generate_chart)
    workflow.add_node("summarizer", summarize_results) # New Summarizer Node
    workflow.add_node("state_printer", state_printer)

    workflow.set_entry_point("preprocess_routing")
    workflow.add_edge("preprocess_routing", "generate_sql")
    workflow.add_edge("generate_sql", "convert_dataframe")

    workflow.add_conditional_edges(
        "convert_dataframe", 
        decide_chart_or_table,
        {
            # Result : Step Name To Go To
            "chart":"instruct_chart_generator", # Path Chart
            "table":"summarizer" # Summarizer
        }
    )

    workflow.add_edge("instruct_chart_generator", "generate_chart")
    workflow.add_edge("generate_chart", "summarizer")
    workflow.add_edge("summarizer", "state_printer") # Add step to summarize the analysis
    workflow.add_edge("state_printer", END)

    app = workflow.compile()

    return app
