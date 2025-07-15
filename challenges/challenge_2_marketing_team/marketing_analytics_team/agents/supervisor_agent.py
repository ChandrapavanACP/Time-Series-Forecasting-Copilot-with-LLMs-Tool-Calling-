# *** MARKETING ANALYTICS SUPERVISOR AGENT

# GOAL: This supervisor agent manages the conversation between the Product Expert, Business Intelligence Expert, and Marketing Email Writer.

# LangChain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import BaseMessage

from langchain_openai import ChatOpenAI

# LangGraph
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict


# * MARKETING ANALYTICS SUPERVISOR

def make_marketing_analytics_supervisor_agent(model, temperature=0):
    
    subagent_names = ["Product_Expert", "Business_Intelligence_Expert", "Marketing_Email_Writer"]
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        llm = ChatOpenAI(model = model)
    else:
        llm = model 
        
    llm.temperature = temperature

    system_prompt = (
        """
        You are a supervisor tasked with managing a conversation between the following workers: {subagent_names}.

        Each worker has the following knowledge and skills:

            1. Product_Expert: Knows course content and can explain details from course sales pages.  
            (Do not have the Product_Expert write emails—that’s the Marketing_Email_Writer’s job.)

            2. Business_Intelligence_Expert: Knows our customer transactions database.  
            Can write SQL, produce tables and charts based on leads, purchases, and transactions data.

            3. Marketing_Email_Writer: Drafts marketing emails using Product_Expert content and  
            customer segments identified by the Business_Intelligence_Expert.

        Assignment Rules:
        - IMPORTANT: **Never** assign the same worker twice in a row unless the worker explicitly requests to continue.  
        - Track which worker acted last and the one previous.  Do not assign the same worker twice in a row.
        - If the same expertise is needed twice, see if a different worker can handle the follow‐up (e.g., Business_Intelligence_Expert hands off to Product_Expert for context).  
        - When multiple workers can fulfill a request, rotate in round‐robin order to balance workload.

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
    
    # LangGraph State Graph
    
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
