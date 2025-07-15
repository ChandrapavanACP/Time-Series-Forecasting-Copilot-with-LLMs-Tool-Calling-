# BUSINESS SCIENCE UNIVERSITY
# PYTHON FOR GENERATIVE AI COURSE
# AI MARKETING ANALYTICS AGENT
# ***

# GOAL: Streamlit chat interface for Marketing Analytics Team
# Tracks: email_subject, email_body, email_list, plots/data, intermediate responses

# Command Line:
#   streamlit run marketing_analytics_team_app_2.py

# LIBRARIES ----
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import warnings
import uuid

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langgraph.checkpoint.memory import MemorySaver

# Add project root directory to sys.path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# -- Constants & Config ------------------------------------------------------
CHAT_LLM_OPTIONS = ["gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-4o"]
EMBEDDING_OPTIONS = ["text-embedding-ada-002"]
PATH_TRANSACTIONS_DB = "sqlite:///challenges/data/database-sql-transactions/leads_scored_segmentation.db"
PATH_PRODUCTS_VDB = "data/data-rag-product-information/products_clean.db"

# -- Marketing Analytics Team Agent 2 ------------------------------------------
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Sequence, TypedDict
import operator
import pandas as pd
from pprint import pprint
from IPython.display import Markdown, Image
import os
import yaml

# Sub-Agents
from marketing_analytics_team.agents.marketing_email_writer_agent import make_marketing_email_writer_agent
from marketing_analytics_team.agents.product_expert import make_product_expert_agent
from marketing_analytics_team.agents.business_intelligence_agent import make_business_intelligence_agent
from marketing_analytics_team.agents.segment_analysis_agent import make_segment_analysis_agent

def make_marketing_analytics_team_2(model, model_embedding, path_products_vector_db, path_transactions_sql_db, checkpointer):
    # API Keys
    os.environ["OPENAI_API_KEY"] = yaml.safe_load(open('../credentials.yml'))['openai']

    # STEP 1: MAKE SUPERVISOR AGENT
    def make_marketing_analytics_supervisor_agent(model, temperature=0):
        subagent_names = [
            "Product_Expert",
            "Business_Intelligence_Expert",
            "Marketing_Email_Writer",
            "Segment_Analysis_Agent"
        ]

        if isinstance(model, str):
            llm = ChatOpenAI(model=model)
        else:
            llm = model

        system_prompt = (
            """
            You are a supervisor tasked with managing a conversation between the following workers: {subagent_names}.

            Each worker has the following knowledge and skills:

                1. Product_Expert: Knows course content and can explain details from course sales pages.
                (Do not have the Product_Expert write emails—that’s the Marketing_Email_Writer’s job.)

                2. Business_Intelligence_Expert: Knows our customer transactions database.
                Can write SQL, produce tables and charts based on leads, purchases, and transactions data.

                3. Marketing_Email_Writer: Drafts marketing emails using Product_Expert content and
                customer segments identified by the Business_Intelligence_Expert or Segment_Analysis_Agent.

                4. Segment_Analysis_Agent: Analyzes precomputed customer segments in the database,
                generating descriptive labels and insights based on lead scores, engagement ratings,
                and purchase frequency.

            Assignment Rules:
            • Track which worker acted last.
            • **Never** assign the same worker twice in a row unless they explicitly request to continue.
            • If the same expertise is needed twice, see if a different worker can handle the follow-up (e.g., Business_Intelligence_Expert hands off to Product_Expert for context).
            • When multiple workers can fulfill a request, rotate in round-robin order to balance workload.

            Workflow:
            1. Read the user’s request.
            2. Decide which worker is best suited **and** is not the same as the last one you chose.
            3. Respond with exactly the worker’s name (e.g. `Business_Intelligence_Expert`) to invoke them.
            4. That worker will perform their task and return results.
            5. Repeat until the task is complete, then respond with `FINISH`.
            """
        )

        route_options = ["FINISH"] + subagent_names

        function_def = {
            "name": "route",
            "description": "Select the next role.",
            "parameters": {
                "title": "route_schema",
                "type": "object",
                "properties": {
                    "next": {
                        "title": "Next",
                        "anyOf": [
                            {"enum": route_options},
                        ],
                    }
                },
                "required": ["next"],
            },
        }

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="messages"),
                (
                    "system",
                    "Given the conversation above, who should act next?"
                    " Or should we FINISH? Select one of: {route_options}",
                ),
            ]
        ).partial(route_options=str(route_options), subagent_names=", ".join(subagent_names))

        supervisor_chain = (
            prompt
            | llm.bind(functions=[function_def], function_call={"name": "route"})
            | JsonOutputFunctionsParser()
        )

        class GraphState(TypedDict):
            messages: Sequence[BaseMessage]
            next: str

        def supervisor_node(state):
            print("---SUPERVISOR---")
            messages = state.get("messages")
            result = supervisor_chain.invoke({"messages": messages})
            next_worker = result.get("next")
            return {"next": next_worker}

        workflow = StateGraph(GraphState)
        workflow.add_node("supervisor", supervisor_node)
        workflow.add_edge(START, "supervisor")
        workflow.add_edge("supervisor", END)
        app = workflow.compile()
        return app

    # STEP 2: INITIALIZE SUB-AGENTS
    supervisor_agent = make_marketing_analytics_supervisor_agent(model=model, temperature=0.7)

    product_expert_agent = make_product_expert_agent(
        model=model,
        model_embedding=model_embedding,
        db_path=path_products_vector_db
    )

    business_intelligence_agent = make_business_intelligence_agent(
        model=model,
        db_path=path_transactions_sql_db
    )

    marketing_agent = make_marketing_email_writer_agent(model=model)

    segment_analysis_agent = make_segment_analysis_agent(
        model=model,
        db_path=path_transactions_sql_db,
        temperature=0.7
    )

    # STEP 3: SUPERVISOR-LED TEAM LANGGRAPH
    class GraphState(TypedDict):
        # Team Messages State Tracking
        messages: Annotated[Sequence[BaseMessage], operator.add]
        # Supervisor State Tracking
        next: str
        # Business Intelligence Expert State Tracking
        sql_query: str
        data: dict
        chart_plotly_code: str
        chart_plotly_json: dict
        # Marketing Email Writer State Tracking
        email_list: list
        email_subject: str
        email_body: str
        # Segment Analysis Agent State Tracking
        segment_labels: dict
        segmentation_data: dict
        chart_json: str

    def supervisor_node(state):
        result = supervisor_agent.invoke(state)
        print(result.get("next"))
        return {'next': result.get("next")}

    def product_expert_node(state):
        result = product_expert_agent.invoke(state)
        return {"messages": result.get("response")}

    def business_intelligence_expert_node(state):
        result = business_intelligence_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "sql_query": result.get("sql_query"),
            "data": result.get("data"),
            "chart_plotly_code": result.get("chart_plotly_code"),
            "chart_plotly_json": result.get("chart_plotly_json"),
        }

    def email_writer_node(state):
        result = marketing_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "email_list": result.get("email_list"),
            "email_subject": result.get("email_subject"),
            "email_body": result.get("email_body"),
        }

    def segment_analysis_node(state):
        result = segment_analysis_agent.invoke(state)
        return {
            "messages": result.get("response"),
            "segment_labels": result.get("segment_labels"),
            "segmentation_data": result.get("segmentation_data"),
            "chart_json": result.get("chart_json"),
        }

    # WORKFLOW DAG
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("Product_Expert", product_expert_node)
    workflow.add_node("Business_Intelligence_Expert", business_intelligence_expert_node)
    workflow.add_node("Marketing_Email_Writer", email_writer_node)
    workflow.add_node("Segment_Analysis_Agent", segment_analysis_node)

    workflow.set_entry_point("supervisor")

    workflow.add_edge('Product_Expert', "supervisor")
    workflow.add_edge('Business_Intelligence_Expert', "supervisor")
    workflow.add_edge('Marketing_Email_Writer', "supervisor")
    workflow.add_edge('Segment_Analysis_Agent', "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state.get("next"),
        {
            'Product_Expert': 'Product_Expert',
            'Business_Intelligence_Expert': 'Business_Intelligence_Expert',
            'Marketing_Email_Writer': 'Marketing_Email_Writer',
            'Segment_Analysis_Agent': 'Segment_Analysis_Agent',
            'FINISH': END
        }
    )

    # Compile with short-term memory
    app = workflow.compile(checkpointer=checkpointer)

    return app

# -- Streamlit Page Setup ---------------------------------------------------
TITLE = "AI Marketing Analytics Team"
st.set_page_config(
    page_title=TITLE,
    page_icon=":bar_chart:",
    layout="wide"
)
st.title(TITLE)

with st.expander("I'm a complete marketing analytics copilot that contains a team of experts: Business intelligence SQL Expert, Product Expert, Marketing email writer, Segment Analysis Agent. (see example questions)"):
    st.markdown(
        """
        #### Business Intelligence
        - Have the Business Intelligence agent show what tables are in the SQL database. Do not engage other agents. 
        - What does the leads_scored table look like?
        - What are the top 10 customers by revenue?
        - Draw a bar chart of the top 10 customers by revenue.
        
        #### Product Expert
        - What are the key features of the 5-Course R-Track?
        - What products are available in the database?
        
        #### Marketing Email Writer
        - Find the top 20 email subscribers ranked by probability of purchase (p1 lead score in the leads_scored table) who have not purchased any courses yet? Have the Product Expert collect information on the 5-Course R-Track for use with the Marketing Expert. Have the Marketing Expert write a compelling marketing email.
        - Have the marketing email writer remove Kamryn Tremblay from the email list
        
        #### Segment Analysis Agent
        - What are the segments in the leads_scored table?
        - Have the Business Intelligence Expert find the top 20 high value buyers from segment 2 (High-Value Customers). Return the top 20 in that segment ranked by p1 lead score who have not purchased Learning Labs PRO product. Do not engage the Product Expert or Marketing Expert.
        - Have the Product Expert collect information on the Learning Labs PRO product for use with the Marketing Expert. Have the Marketing Expert write a compelling marketing email to the 20 high value buyers from the previous analysis. Don't engage the Business Intelligence Expert or the Segment Agent.
        """
    )

# -- Sidebar: Controls & Summary ---------------------------------------------
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox("OpenAI Model", CHAT_LLM_OPTIONS)
    embed_option = st.selectbox("Embedding Model", EMBEDDING_OPTIONS)
    add_short_term_memory = st.checkbox("Add Short-Term Memory", value=True)
    show_reasoning = True
    if add_short_term_memory:
        if st.button("Clear Chat History"):
            msgs = StreamlitChatMessageHistory(key="marketing_messages")
            msgs.clear()
            msgs.add_ai_message("How can I help with your marketing analytics today?")
            st.session_state.details = []
            st.session_state.checkpointer = MemorySaver()
    else:
        st.session_state.checkpointer = None

# -- Cache the Checkpointer --------------------------------------------------
@st.cache_resource
def get_checkpointer():
    """Initialize and cache the LangGraph MemorySaver checkpointer."""
    return MemorySaver()

# -- Initialize Session State -----------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # Unique thread ID per session
if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = get_checkpointer()  # Cache checkpointer
if "details" not in st.session_state:
    st.session_state.details = []

# -- Initialize History & Detail Store --------------------------------------
msgs = StreamlitChatMessageHistory(key="marketing_messages")
if not msgs.messages:
    msgs.add_ai_message("How can I help with your marketing analytics today?")

# -- Function to Render Chat ------------------------------------------------
def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        # Skip initial greeting if there are newer messages
        if i == 0 and msg.content == "How can I help with your marketing analytics today?" and len(msgs.messages) > 1:
            continue
        with st.chat_message(msg.type):
            content = msg.content
            if content.startswith("DETAILS_INDEX:"):
                idx = int(content.split(":")[1])
                detail = st.session_state.details[idx]
                with st.expander("Marketing Analysis", expanded=True):
                    tabs = st.tabs(["AI Reasoning", "SQL Query", "Plot/Data", "Email & List", "Segmentation"])
                    with tabs[0]:
                        text = detail.get("reasoning", "_(No reasoning available)_")
                        if show_reasoning:
                            st.write(text)
                        else:
                            st.info("Enable 'Show AI Reasoning' in sidebar to view intermediate thoughts.")
                    with tabs[1]:
                        sql = detail.get("sql_query")
                        if sql:
                            st.code(sql, language="sql")
                        else:
                            st.info("No SQL query generated.")
                    with tabs[2]:
                        if detail.get("chart_json"):
                            fig = pio.from_json(detail["chart_json"])
                            st.plotly_chart(fig, key=f"plot_data_{i}")  # Unique key
                        elif detail.get("data") is not None:
                            df = pd.DataFrame(detail["data"])
                            st.dataframe(df)
                        else:
                            st.info("No plot or data returned.")
                    with tabs[3]:
                        st.markdown("**Subject:**")
                        st.write(detail.get("title", ""))
                        st.markdown("**Body:**")
                        st.write(detail.get("body", ""))
                        st.markdown("**Recipients:**")
                        st.write(detail.get("list", []))
                    with tabs[4]:
                        if detail.get("segmentation_chart_json"):
                            fig = pio.from_json(detail["segmentation_chart_json"])
                            st.plotly_chart(fig, key=f"segmentation_{i}")  # Unique key
                        if detail.get("segmentation_data") is not None:
                            df = pd.DataFrame(detail["segmentation_data"])
                            st.dataframe(df)
                        else:
                            st.info("No segmentation data returned.")
            else:
                st.write(content)

# -- Initialize LLM & Team --------------------------------------------------
llm = ChatOpenAI(model=model_option)
embeddings = OpenAIEmbeddings(model=embed_option)

marketing_team = make_marketing_analytics_team_2(
    model=llm,
    model_embedding=embeddings,
    path_products_vector_db=PATH_PRODUCTS_VDB,
    path_transactions_sql_db=PATH_TRANSACTIONS_DB,
    checkpointer=st.session_state.checkpointer,
)

# -- Handle User Input & AI Response ----------------------------------------
if prompt := st.chat_input("Enter your marketing analytics request here…"):
    with st.spinner("Thinking..."):
        # Add user message
        st.chat_message("human").write(prompt)
        msgs.add_user_message(prompt)

        try:
            result = marketing_team.invoke(
                input={
                    "messages": [HumanMessage(content=prompt)]
                },
                config={
                    "recursion_limit": 10,
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
            )
        except Exception as e:
            st.error("Error invoking marketing team.")
            print(e)
            result = None

    if result:
        recipients = result.get("email_list", [])

        # Collect reasoning with agent names, only for AI messages after the latest Human message
        reasoning = ""
        latest_human_index = -1
        for i, message in enumerate(result.get("messages", [])):
            if isinstance(message, HumanMessage):
                latest_human_index = i
        for message in result.get("messages", [])[latest_human_index + 1:]:
            if isinstance(message, AIMessage):
                reasoning += f"##### {message.name}:\n\n{message.content}\n\n---\n\n"

        # Collect detail
        detail = {
            "reasoning": reasoning,
            "sql_query": result.get("sql_query"),
            "chart_json": result.get("chart_plotly_json"),
            "data": result.get("data"),
            "title": result.get("email_subject"),
            "body": result.get("email_body"),
            "list": recipients,
            "segmentation_chart_json": result.get("chart_json"),
            "segmentation_data": result.get("segmentation_data"),
        }
        idx = len(st.session_state.details)
        st.session_state.details.append(detail)
        msgs.add_ai_message(f"DETAILS_INDEX:{idx}")

# Render current messages from StreamlitChatMessageHistory
display_chat_history()
