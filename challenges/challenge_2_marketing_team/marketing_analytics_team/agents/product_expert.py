# *** PRODUCT EXPERT (BASED ON RAG AGENT FROM CLINIC #1) ***

# Key Modifications:
# 1. Integrates a Vector Database (Chroma) to retrieve product information.
# 2. Implements Messages History "chat_history" to provide context for the agent based on a sequence of messages.
# 3. Returns a Compiled LangGraph app

# LIBRARIES

from langchain_chroma import Chroma

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, AIMessage

from langchain.prompts import PromptTemplate

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

from langgraph.graph import StateGraph, START, END

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from typing import Sequence, TypedDict

from marketing_analytics_team.agents.utils import get_last_human_message


# KEY INPUTS 
PATH_PRODUCTS_VECTORDB = "data/data-rag-product-information/products_clean.db"


# * AGENT CREATION
def make_product_expert_agent(
    model, model_embedding='text-embedding-ada-002', db_path=PATH_PRODUCTS_VECTORDB
):
    """
    Create a Product Expert Agent that can answer questions about products.
    
    Parameters:
    ---------
        - model: The language model to use for the agent.
        - db_path: Path to the vector database containing product information.
    
    Returns:
    -------
        - app: The compiled agent workflow.
    """    
    
    # Handle case when users want to make a different model than ChatOpenAI
    if isinstance(model, str):
        model = ChatOpenAI(model = model)
        
    if isinstance(model_embedding, str):
        embedding_function = OpenAIEmbeddings(
            model=model_embedding,
        )
    else:
        embedding_function = model_embedding
    
    
    # * CREATE AGENT COMPONENTS
    
    # Preprocessor Agent
    def create_rag_question_preprocessor_agent(llm, temperature):
        
        llm.temperature = temperature
        
        prompt = PromptTemplate(
            template="""
            You are a question preparer for a Product Expert. Your goal is to extract the relevant part of the question so the Product Expert knows which product or products to provide information on. 
            
            Remove anything about writing marketing emails or emails in general. 
            
            Remove anything related to business analytics that requires knowledge of a customers, transactions, or subscribers. Leave only information related to collecting information on the product or products in question. 
            
            Only return the product or products to collect information on, and what information to collect on those products. 
            
            User Input: {user_question}
            """,
            input_variables=['user_question']
        )
        
        rag_preprocessor = prompt | llm | StrOutputParser()
        
        return rag_preprocessor
    
    rag_preprocessor = create_rag_question_preprocessor_agent(llm=model, temperature=0)
    
    # RAG Agent
    def create_rag_agent(db_path, llm, temperature):
        
        llm.temperature = temperature
        
        vectorstore = Chroma(
            embedding_function=embedding_function, 
            persist_directory=db_path
        )

        retriever = vectorstore.as_retriever()
        
        contextualize_q_system_prompt = """Given a chat history and the latest user question \
        which might reference context in the chat history, formulate a standalone question \
        which can be understood without the chat history. Do NOT answer the question, \
        just reformulate it if needed and otherwise return it as is."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
        
        qa_system_prompt = """You are an assistant for question-answering tasks. \
        Use the following pieces of retrieved context to answer the question. \
        If you don't know the answer, just say that you don't know. \

        {context}"""
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        return rag_chain

    rag_agent = create_rag_agent(db_path=db_path, llm=model, temperature=0.7)
    
    # * CREATE AGENT LANGGRAPH WORKFLOW
    
    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        response: Sequence[BaseMessage]
        
    def product_expert_node(state):
    
        print("---PRODUCT EXPERT---")
        
        # print(state["messages"])
        
        messages = state.get("messages")
        
        last_question = get_last_human_message(messages)
        if last_question:
            last_question = last_question.content
        
        # Implement a preprocessor
        last_question = rag_preprocessor.invoke({'user_question': last_question})
        
        result = rag_agent.invoke({"input": last_question, "chat_history": messages})
        
        # print(result)
        
        return {
            "response": [AIMessage(content=result['answer'], name='Product_Expert')],
        }
    
    workflow = StateGraph(GraphState)

    workflow.add_node("product_expert", product_expert_node)
    
    workflow.add_edge(START, "product_expert")
    workflow.add_edge("product_expert", END)
    
    app = workflow.compile()
    
    return app
