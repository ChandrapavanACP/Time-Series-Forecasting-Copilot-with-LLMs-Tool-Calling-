# BUSINESS SCIENCE
# This file is part of the Business Science Customer Segmentation Agent Challenge.

from typing import Dict, Any, Sequence
import pandas as pd
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from typing import Sequence, TypedDict
import plotly.express as px
from marketing_analytics_team.agents.utils import get_last_human_message

db_path = "sqlite:///challenges/data/database-sql-transactions/leads_scored_segmentation.db"


def make_segment_analysis_agent(model, db_path, temperature=0):
    # Handle model initialization
    if isinstance(model, str):
        llm = ChatOpenAI(model=model)
    else:
        llm = model
    llm.temperature = temperature

    # Segment Analysis Prompt
    segment_analysis_prompt = PromptTemplate(
        template="""
        You are an expert in marketing analytics for Business Science, a premium data science educational platform.
        Analyze the user's request to determine if it requires analyzing customer segments from the database.

        The segments are precomputed with numeric IDs (e.g., 0, 1, 2). Your task is to:
        1. Generate descriptive labels for each segment based on their statistics.
        2. Provide insights into patterns across segments. Call this section "Segment Insights".
        3. Suggest detailed marketing implications and campaign strategies for the different segments based on their attributes. Call this section "Marketing Implications".

        Metrics provided:
        - avg_p1: Lead score (0 to 1, higher means more likely to purchase).
        - avg_member_rating: Engagement rating (1 to 5, higher means more engaged).
        - avg_purchase_frequency: Average number of transactions per customer.
        - customer_count: Number of customers in the segment.

        If segment analysis is requested, provide:
        1. A general response summarizing the analysis.
        2. A dictionary mapping segment IDs to descriptive labels (e.g., {{"0": "High-Value Customers"}}).
        3. Detailed insights explaining patterns and marketing implications.
        4. A summary table of segment statistics in markdown format, using the generated labels.
        
        In the general response, include:
        - A summary of the analysis.
        - Provide insights into patterns across segments. Call this section "Segment Insights".
        - Suggest campaign strategies for the different segments. Title this as "Marketing Implications".
        - Use bullets and tables to make the response clear and easy to read.

        RETURN FORMAT:
        A strict JSON object (check to make sure it is valid JSON) with the following:
        - If analysis is requested:
        {{
            "general_response": "Summary of the analysis",
            "analysis_required": true,
            "segment_labels": {{ "0": "Label for segment 0", "1": "Label for segment 1", "2": "Label for segment 2" }},
            "insights": "Detailed explanation of patterns and marketing implications",
            "summary_table": "Markdown table with segment_name (use generated labels), avg_p1, avg_member_rating, avg_purchase_frequency, customer_count"
        }}
        - If no analysis is required:
        {{
            "general_response": "Response indicating no segment analysis needed",
            "analysis_required": false,
            "segment_labels": {{}},
            "insights": "",
            "summary_table": ""
        }}

        INITIAL_USER_QUESTION: {initial_question}
        CONTEXT: {chat_history}
        SEGMENT_STATISTICS: {segment_statistics}
        """,
        input_variables=["initial_question", "chat_history", "segment_statistics"]
    )

    segment_analyzer = segment_analysis_prompt | llm | JsonOutputParser()

    # Graph State
    class GraphState(TypedDict):
        messages: Sequence[BaseMessage]
        response: Sequence[BaseMessage]
        insights: str
        summary_table: str
        analysis_required: bool
        segment_labels: dict
        segmentation_data: dict
        chart_json: str

    def segment_analysis_node(state: GraphState) -> GraphState:
        print("---SEGMENT ANALYSIS AGENT---")

        # Connect to database
        engine = create_engine(db_path)
        conn = engine.connect()
        leads_query = """
        SELECT user_email, p1, member_rating, segment
        FROM leads_scored
        """
        transactions_query = """
        SELECT user_email, purchased_at
        FROM transactions
        """
        df_leads = pd.read_sql(leads_query, conn)
        df_transactions = pd.read_sql(transactions_query, conn)
        conn.close()

        # Calculate purchase frequency
        purchase_freq = df_transactions.groupby("user_email").size().reset_index(name="purchase_frequency")

        # Merge data
        df_analysis = df_leads.merge(purchase_freq, on="user_email", how="left")
        df_analysis["purchase_frequency"] = df_analysis["purchase_frequency"].fillna(0)

        # Compute summary statistics
        df_summary = df_analysis.groupby("segment").agg({
            "p1": "mean",
            "member_rating": "mean",
            "purchase_frequency": "mean",
            "user_email": "count"
        }).rename(columns={"user_email": "customer_count"}).reset_index()

        # Round metrics for display
        df_summary["avg_p1"] = df_summary["p1"].round(3)
        df_summary["avg_member_rating"] = df_summary["member_rating"].round(2)
        df_summary["avg_purchase_frequency"] = df_summary["purchase_frequency"].round(2)

        # Prepare segment statistics for LLM
        segment_stats_json = df_summary[["segment", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]].to_json(orient="records")
        print("Segment Statistics JSON:", segment_stats_json)

        # Invoke LLM for labels and insights
        messages = state.get("messages")
        last_question = get_last_human_message(messages)
        last_question = last_question.content if last_question else ""
        result = segment_analyzer.invoke({
            "initial_question": last_question,
            "chat_history": messages,
            "segment_statistics": segment_stats_json
        })
        print("LLM Result:", result)

        # Validate and apply LLM-generated labels
        default_labels = {str(i): f"Segment {i}" for i in df_summary["segment"]}
        segment_labels = result.get("segment_labels", default_labels)
        # Ensure segment IDs are strings and validate labels
        segment_labels = {str(k): v for k, v in segment_labels.items() if isinstance(v, str) and v}
        if not all(str(seg) in segment_labels for seg in df_summary["segment"]):
            print("Warning: Missing labels for some segments. Using defaults.")
            segment_labels = default_labels

        # Apply labels
        df_summary["segment_name"] = df_summary["segment"].astype(str).map(segment_labels)

        # Create summary table with LLM-generated labels
        if result["analysis_required"]:
            summary_table = df_summary[["segment_name", "segment", "avg_p1", "avg_member_rating", "avg_purchase_frequency", "customer_count"]].to_markdown(index=False)
            # Update result with the correct summary table
            result["summary_table"] = summary_table
        else:
            summary_table = ""

        # Create Plotly bar chart
        df_viz = df_summary.melt(
            id_vars=["segment_name"],
            value_vars=["avg_p1", "avg_member_rating", "avg_purchase_frequency"],
            var_name="metric",
            value_name="value"
        )
        fig = px.bar(
            df_viz,
            x="segment_name",
            y="value",
            color="metric",
            barmode="group",
            title="Segment Analysis: Lead Score, Member Rating, and Purchase Frequency",
            labels={"segment_name": "Segment", "value": "Average Value", "metric": "Metric"}
        )
        chart_json = fig.to_json()

        # Format response
        if result["analysis_required"]:
            formatted_response = (
                f"**General Response**: {result['general_response']}\n\n"
                f"**Insights**: {result['insights']}\n\n"
                f"**Summary Table**:\n{result['summary_table']}"
            )
        else:
            formatted_response = (
                f"**General Response**: {result['general_response']}\n\n"
                "No segment analysis required."
            )

        return {
            "response": [AIMessage(content=formatted_response, name="SegmentAnalysisAgent")],
            "insights": result.get("insights", ""),
            "summary_table": result.get("summary_table", ""),
            "analysis_required": result.get("analysis_required", False),
            "segment_labels": segment_labels,
            "segmentation_data": df_summary.to_dict(),
            "chart_json": chart_json
        }

    # Create LangGraph workflow
    workflow = StateGraph(GraphState)
    workflow.add_node("segment_analyzer", segment_analysis_node)
    workflow.add_edge(START, "segment_analyzer")
    workflow.add_edge("segment_analyzer", END)
    app = workflow.compile()
    
    return app

if __name__ == "__main__":
    
    from langchain_core.messages import HumanMessage
    from pprint import pprint
    from IPython.display import Markdown
    
    # Example usage
    model = "gpt-4.1-nano"
    db_path = "sqlite:///challenges/data/database-sql-transactions/leads_scored_segmentation.db"
    temperature = 0.7

    agent = make_segment_analysis_agent(model, db_path, temperature)
    
    messages = [HumanMessage(content="Can you analyze the segments and provide insights?")]
    
    results = agent.invoke({"messages": messages})
    
    results.keys()
    
    Markdown(results['response'][0].content)
    
    # Display the chart
    import plotly.io as pio
    fig = pio.from_json(results['chart_json'])
    fig
    
    # Display Summary Table
    Markdown(results['summary_table'])
    