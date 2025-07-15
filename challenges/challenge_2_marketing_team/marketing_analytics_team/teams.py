# *** AI MARKETING ANALYTICS TEAM USING A SUPERVISOR ARCHITECTURE ***
# GENERATIVE AI BOOTCAMP BY BUSINESS SCIENCE
# ***



# LIBRARIES

# LangChain
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LangGraph
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
import operator

# Sub-agents
from marketing_analytics_team.agents.business_intelligence_agent import make_business_intelligence_agent
from marketing_analytics_team.agents.marketing_email_writer_agent import make_marketing_email_writer_agent
from marketing_analytics_team.agents.product_expert import make_product_expert_agent
from marketing_analytics_team.agents.supervisor_agent import make_marketing_analytics_supervisor_agent

# Add Short Term Memory
from langgraph.checkpoint.memory import MemorySaver

# * TEAM CREATION

def make_marketing_analytics_team(model, model_embedding, path_products_vector_db, path_transactions_sql_db, checkpointer = None):
    """
    Create a AI Marketing Analytics Team with the following sub-agents:
    1. Product Expert Agent
    2. Business Intelligence Agent
    3. Marketing Email Writer Agent
    4. Supervisor Agent
    
    Parameters:
    ----------
        - model: The language model to use for the agents.
        - model_embedding: The embedding model to use for the agents.
        - path_products_vector_db: Path to the vector database containing product information.
        - path_transactions_sql_db: Path to the SQL database containing transaction data.
        - checkpointer: Optional checkpointer for short term memory.
    
    Returns:
    -------
        - app: The compiled agent workflow.
    """
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
        
    if isinstance(model_embedding, str):
        embedding_function = OpenAIEmbeddings(
            model=model_embedding,
        )
    else:
        embedding_function = model_embedding
        
    # * CREATE SUPERVISOR
        
    supervisor_agent = make_marketing_analytics_supervisor_agent(model=llm, temperature=0.7)
        
    # * CREATE SUB AGENTS
    
    product_expert_agent = make_product_expert_agent(model=llm, model_embedding=embedding_function, db_path=path_products_vector_db)
    business_intelligence_agent = make_business_intelligence_agent(model=llm, db_path=path_transactions_sql_db)
    marketing_agent = make_marketing_email_writer_agent(model=llm)
    
    # * LANGGRAPH

    class GraphState(TypedDict):
        # Team Messages State Tracking
        messages: Annotated[Sequence[BaseMessage], operator.add]
        # Supervisor State Tracking
        next: str
        # Business Intelligence Expert State Tracking
        sql_query : str
        data: dict
        chart_plotly_code: str
        chart_plotly_json: dict
        # Marketing Email Writer State Tracking
        email_list: list
        email_subject: str
        email_body: str
        

        
    def supervisor_node(state):
        
        result = supervisor_agent.invoke(state)
        
        print(result.get("next"))
        
        return {'next': result.get("next")}


    def product_expert_node(state):
        
        result = product_expert_agent.invoke(state)
        
        return {
            "messages": result.get("response"),
        }
        
    def business_intelligence_expert_node(state):
        
        result = business_intelligence_agent.invoke(state)
        
        return {
            "messages": result.get("response"),
            # Business Intelligence Expert State Tracking
            "sql_query": result.get("sql_query"),
            "data": result.get("data"),
            "chart_plotly_code": result.get("chart_plotly_code"),
            "chart_plotly_json": result.get("chart_plotly_json"),
        }


    def email_writer_node(state):
        
        result = marketing_agent.invoke(state)
        
        return {
            "messages": result.get("response"),
            # Marketing Email Writer State Tracking
            "email_list": result.get("email_list"),
            "email_subject": result.get("email_subject"),
            "email_body": result.get("email_body"),
        }

    # * WORKFLOW DAG

    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Product_Expert", product_expert_node)
    workflow.add_node("Business_Intelligence_Expert", business_intelligence_expert_node)
    workflow.add_node("Marketing_Email_Writer", email_writer_node)

    workflow.set_entry_point("supervisor")

    workflow.add_edge('Product_Expert', "supervisor")
    workflow.add_edge('Business_Intelligence_Expert', "supervisor")
    workflow.add_edge('Marketing_Email_Writer', "supervisor")

    workflow.add_conditional_edges(
        "supervisor", 
        lambda state: state.get("next"), 
        {
            'Product_Expert': 'Product_Expert', 
            'Business_Intelligence_Expert': 'Business_Intelligence_Expert', 
            'Marketing_Email_Writer':'Marketing_Email_Writer',
            'FINISH': END
        }
    )

    # * NEW: Short Term Memory
    app = workflow.compile(checkpointer=checkpointer)

    return app
    
    

