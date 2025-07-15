# *** MARKETING COPY-WRITER / EMAIL EXPERT (BASED ON PROMPT ENGINEERING FROM AI FAST TRACK) ***

# Key Modifications:
# 1. Implements Prompt Template and prompt engineering to create a marketing email copywriter persona.
# 2. Integrates Messages History ("chat_history") to provide context for the agent based on the previous sequence of messages.
# 3. Implements Structured Outputs if an Email is needed.
# 4. Returns a Compiled LangGraph app

# LIBRARIES

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.messages import BaseMessage, AIMessage

from langgraph.graph import StateGraph, START, END

from typing import Sequence, TypedDict

from marketing_analytics_team.agents.utils import get_last_human_message

# AGENT

def make_marketing_email_writer_agent(model, temperature=0):
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model
        
    llm.temperature = temperature
    
    # * Marketing Agent

    marketing_agent_prompt = PromptTemplate(
        template="""
        You are an expert in writing marketing email copy for Business Science, a premium data science educational platform. 
        Analyze the user's request and determine if it requires sending a marketing email.
        
        Your emails are designed to inform customers about products that they might be interested in, and to target customers by email address.
        
        Your email should specify:
        
        1. Which customer email addresses to target
        2. An attention grabbing email subject
        3. Include email content designed to educate and provide reasons that they should invest in our educational program
        
        Examples of good reasons to purchase include:
        1. Getting a career advancement or new job
        2. Developing a portfolio to attract recruiters and wow their bosses
        3. Taking action: Stop procrastinating and finally take action towards a fullfilling career
        4. Increased Salary: Many Data Scientists make over $100,000 per year. 
        
        IMPORTANT: Make sure to explain why the product you are recommending will help them address their goal.
        
        EMAIL LIST: Generally this will come from a Business Intelligence Agent. Do NOT make up email addresses. Just return an empty list [] if you don't know.
        
        RETURN FORMAT:
        Output must be a strict JSON object. Do NOT include comments or trailing commas. 
        Do NOT explain anything outside the JSON block.

        - If an email is requested, respond ONLY with:
        {{
            "general_response": "Your general response to the user question"
            "email_required": true,
            "email_list": ["list_of_target_emails"] or [] (if you were not provided with email addresses via the Business Intelligence Agent),
            "email_subject": "Your compelling email subject",
            "email_body": "Your detailed email body content",            
        }}

        - If no email is required, respond ONLY with:
        {{
            "general_response": "Your general response to the user question"
            "email_required": false,
            "email_list": [],
            "email_subject": "",
            "email_body": ""            
        }}

        INITIAL_USER_QUESTION: {initial_question}
        
        CONTEXT: {chat_history}
        """,
        input_variables=["initial_question", "chat_history"]
    )

    marketing_agent = marketing_agent_prompt | llm | JsonOutputParser()
    
    # * CREATE AGENT LANGGRAPH WORKFLOW
    
    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        response: Sequence[BaseMessage]
        email_list: list
        email_subject: str
        email_body: str
        email_required: bool
        
    def marketing_email_writer_node(state):

        print("---MARKETING EMAIL WRITER---")

        messages = state.get("messages")
        last_question = get_last_human_message(messages)
        last_question = last_question.content if last_question else ""

        # Structured JSON output from the agent
        result = marketing_agent.invoke({
            'initial_question': last_question,
            'chat_history': messages
        })

        if result["email_required"]:
            # Format the email if required
            formatted_response = (
                f"**General Response**: {result['general_response']}\n\n"
                "---\n\n"
                f"**Email Targets**: {', '.join(result['email_list'])}\n\n"
                f"**Subject**: {result['email_subject']}\n\n"
                f"{result['email_body']}"
            )
        else:
            # Clear and friendly fallback if no email is requested
            formatted_response = (
                f"**General Response**: {result['general_response']}\n\n"
                "No email is required."
            )

        return {
            "response": [AIMessage(content=formatted_response, name='Marketing_Email_Writer')],
            # Structured output for email details
            "email_list": result.get("email_list", []),
            "email_subject": result.get("email_subject", ""),
            "email_body": result.get("email_body", ""),
            "email_required": result.get("email_required", False)
        }
    
    workflow = StateGraph(GraphState)

    workflow.add_node("marketing_expert", marketing_email_writer_node)
    
    workflow.add_edge(START, "marketing_expert")
    workflow.add_edge("marketing_expert", END)
    
    app = workflow.compile()
    
    return app



