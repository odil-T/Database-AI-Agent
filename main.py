import operator
import re
import sqlite3
from typing import TypedDict, Annotated, Literal
from uuid import uuid4

import pandas as pd
from cohere import ToolMessage
from langchain.messages import SystemMessage, HumanMessage, AIMessage, AnyMessage
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from chroma_vectorstore import get_vectorstore
from config import COHERE_API_KEY

vectorstore = get_vectorstore()
chat_model = ChatCohere(cohere_api_key=COHERE_API_KEY)

FORBIDDEN_SQL_CLAUSES = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "RENAME",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK"
]


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: HumanMessage
    retrieved_table_schemas: list[Document]
    relevant_table_schemas: list[str]
    sql_query: str
    feedback_on_sql_query: str
    sql_query_output: str


def save_user_query(state: State):
    i = 1
    while True:
        user_query = state["messages"][-i]
        if type(user_query) == HumanMessage:
            break
        i += 1
    return {"user_query": user_query}


def is_user_query_viable(state: State) -> Literal["retrieve_table_schemas", "invalid_user_query_node"]:
    prompt = f"""You are a classifier for a database assistant.

Your task is to determine whether a user's message is a
NATURAL LANGUAGE QUERY that can reasonably be translated into a SQL query.

A message SHOULD be classified as SQL_QUERY if:
- It asks to retrieve, filter, aggregate, compare, or sort data
- It refers to tables, records, metrics, time ranges, counts, averages, sums, etc.
- It could be answered by querying a relational database

A message SHOULD be classified as NOT_SQL_QUERY if:
- It is casual conversation, greeting, or small talk
- It asks for explanations, definitions, opinions, or advice
- It asks how to do something (procedural / instructional)
- It asks to write code, prompts, or documentation
- It is ambiguous or lacks any data-related intent

Do NOT assume missing details.
If a SQL query cannot be reasonably inferred, classify as NOT_SQL_QUERY.

Return ONLY one of the following labels:
- SQL_QUERY
- NOT_SQL_QUERY

--------------------

Here is the user's message:
{state["user_query"].content}
"""
    ai_response = chat_model.invoke([HumanMessage(content=prompt)]).content
    if "SQL_QUERY" == ai_response.upper().strip():
        return "retrieve_table_schemas"
    else:
        return "invalid_user_query_node"


def invalid_user_query_node(state: State):
    return {"messages": [AIMessage(content="Your request is not a viable query.")]}


def retrieve_table_schemas(state: State):
    retrieved_table_schemas = vectorstore.max_marginal_relevance_search(
        query=state["user_query"].content,
        k=10,
        fetch_k=30,
        lambda_mult=0.4
    )
    return {"retrieved_table_schemas": retrieved_table_schemas}


def filter_table_schemas_with_llm(state: State):
    retrieved_table_schemas_str = ""
    for doc in state["retrieved_table_schemas"][:-1]:
        retrieved_table_schemas_str += "### " + doc.page_content + "\n\n\n"
    retrieved_table_schemas_str += "### " + state["retrieved_table_schemas"][-1].page_content

    filter_prompt = f"""You are a strict classifier.

Your task is to identify ALL database tables whose schemas are relevant to the user's query.
There is NO limit on the number of relevant tables. Select every table that could reasonably be needed to answer the query.

IMPORTANT OUTPUT RULES:
- Output MUST be a single line
- Output MUST be a comma-separated list of table names
- Use table names EXACTLY as they appear in the schemas
- Do NOT include explanations, comments, or extra text
- If NO tables are relevant, output an empty string

------------------------
User query:
{state["user_query"]}
------------------------

------------------------
Table schemas:
{retrieved_table_schemas_str}
------------------------"""

    relevant_table_schemas = chat_model.invoke([HumanMessage(content=filter_prompt)]).content.split(", ")
    return {"relevant_table_schemas": relevant_table_schemas}


def generate_sql_with_llm(state: State):
    if not state["feedback_on_sql_query"]:
        generate_sql_prompt = f"""You are an SQL expert. You will be given a user query and relevant table schemas. Your task is to generate a MySQL query that answers the user's query. Adhere strictly to the rules below when generating the query.

SAFETY RULES (MANDATORY):
- Generate ONLY a single SELECT statement
- NEVER generate {", ".join(FORBIDDEN_SQL_CLAUSES)}
- Do NOT use multiple statements or semicolons
- Use ONLY tables and columns provided in the schema
- Do NOT invent schema elements
- Do NOT use SELECT *
- Always include a `LIMIT 200` clause unless explicitly told otherwise
- Output ONLY the SQL query, no explanations or comments
- If the query cannot be answered using the schema, output: CANNOT_ANSWER_WITH_GIVEN_SCHEMA
- SQL dialect: MySQL

------------------------
User query:
{state["user_query"].content}
------------------------

------------------------
Table schemas:
{state["relevant_table_schemas"]}
------------------------"""
    else:
        generate_sql_prompt = f"""You are an SQL expert. You will be given a user query, relevant table schemas, a sample SQL query that tries to answer the user's query, and feedback on why the sample SQL query is incorrect.
Your task is to correct the sample SQL query based on the given feedback such that it answers the user's query with the relevant table schemas. Adhere strictly to the rules below when correcting the query.

SAFETY RULES (MANDATORY):
- Generate ONLY a single SELECT statement
- NEVER generate {", ".join(FORBIDDEN_SQL_CLAUSES)}
- Do NOT use multiple statements or semicolons
- Use ONLY tables and columns provided in the schema
- Do NOT invent schema elements
- Do NOT use SELECT *
- Always include a `LIMIT 50` clause unless explicitly told otherwise
- Output ONLY the SQL query, no explanations or comments
- If the query cannot be answered using the schema, output: CANNOT_ANSWER_WITH_GIVEN_SCHEMA
- SQL dialect: MySQL

------------------------
User query:
{state["user_query"].content}
------------------------

------------------------
Table schemas:
{state["relevant_table_schemas"]}
------------------------

------------------------
Incorrect SQL query:
{state["sql_query"]}
------------------------

------------------------
Feedback on incorrect SQL query:
{state["feedback_on_sql_query"]}
------------------------"""

    sql_query = chat_model.invoke([HumanMessage(content=generate_sql_prompt)]).content
    return {"sql_query": sql_query}


def give_feedback_on_sql_query(state: State):
    sql_query = state["sql_query"]

    if re.search(r"(SELECT.*\*[^/])", sql_query, re.IGNORECASE):
        return {"feedback_on_sql_query": "`SELECT *` is not allowed. Select only the necessary columns."}

    found_forbidden_sql_clauses = [clause for clause in FORBIDDEN_SQL_CLAUSES if clause in sql_query.upper().split()]
    if found_forbidden_sql_clauses:
        return {"feedback_on_sql_query": f"The following forbidden SQL clauses were found: {", ".join(found_forbidden_sql_clauses)}. They should not exist in the SQL query."}

    if sql_query.count(";") > 1 or "\n\n" in sql_query:
        return {"feedback_on_sql_query": "More than one SQL statement was found. Only one SQL statement is allowed."}

    return {"feedback_on_sql_query": ""}


def is_sql_query_safe(state: State) -> Literal["generate_sql_with_llm", "execute_sql_query"]:
    if state["feedback_on_sql_query"]:
        return "generate_sql_with_llm"
    return "execute_sql_query"


def execute_sql_query(state: State):
    try:
        with sqlite3.connect("my_database.db") as conn:
            df = pd.read_sql_query(state["sql_query"], conn)
            sql_query_output = df.to_csv(index=False)
    except Exception as e:
        sql_query_output = e
    finally:
        return {"sql_query_output": sql_query_output}


def llm_call(state: State):
    system_prompt = """You are a data analyst assistant.
Answer the user's question using ONLY the provided database results.
If the data is insufficient, say so explicitly.
If there is an error in the database's results, mention it.
Do not invent numbers."""
    messages_to_invoke = [SystemMessage(content=system_prompt)] + state["messages"] + [ToolMessage(content=state["sql_query_output"], tool_call_id=uuid4())]
    ai_response = chat_model.invoke(messages_to_invoke)
    return {"messages": [ai_response]}


agent_builder = StateGraph(State)

agent_builder.add_node("save_user_query", save_user_query)
agent_builder.add_node("is_user_query_viable", is_user_query_viable)
agent_builder.add_node("invalid_user_query_node", invalid_user_query_node)
agent_builder.add_node("retrieve_table_schemas", retrieve_table_schemas)
agent_builder.add_node("filter_table_schemas_with_llm", filter_table_schemas_with_llm)
agent_builder.add_node("generate_sql_with_llm", generate_sql_with_llm)
agent_builder.add_node("give_feedback_on_sql_query", give_feedback_on_sql_query)
agent_builder.add_node("is_sql_query_safe", is_sql_query_safe)
agent_builder.add_node("execute_sql_query", execute_sql_query)
agent_builder.add_node("llm_call", llm_call)

agent_builder.add_edge(START, "save_user_query")
agent_builder.add_conditional_edges(
    "save_user_query",
    "is_user_query_viable",
    ["retrieve_table_schemas", "invalid_user_query_node"]
)
agent_builder.add_edge("invalid_user_query_node", END)
agent_builder.add_edge("retrieve_table_schemas", "filter_table_schemas_with_llm")
agent_builder.add_edge("filter_table_schemas_with_llm", "generate_sql_with_llm")
agent_builder.add_edge("generate_sql_with_llm", "give_feedback_on_sql_query")
agent_builder.add_conditional_edges(
    "give_feedback_on_sql_query",
    "is_sql_query_safe",
    ["generate_sql_with_llm", "execute_sql_query"]
)
agent_builder.add_edge("execute_sql_query", "llm_call")
agent_builder.add_edge("llm_call", END)


checkpointer = InMemorySaver()
agent = agent_builder.compile(checkpointer=checkpointer)
config = {"configurable": {"thread_id": "1"}}


# TODO - we should just include the latest user message manually when invoking the agent; the agent returns the final checkpoint's state, i.e. the full messages history
# user_query = [HumanMessage(content="Which users have over 1000 attendances?")]
# final_state = agent.invoke({"messages": user_query}, config)
# for m in final_state["messages"]:
#     m.pretty_print()
