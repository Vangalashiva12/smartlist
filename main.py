from dotenv import load_dotenv
import os
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic_core.core_schema import model_field
from langchain.tools import tool
from langchain.agents import create_openai_tools_agent, AgentExecutor
from todoist_api_python.api import TodoistAPI

load_dotenv()
todoist_api_key= os.getenv("TODOIST_API_KEY")
gemini_api_key= os.getenv("GEMINI_API_KEY")
todoist = TodoistAPI(todoist_api_key)

@tool
def add_task(task,details=None):
    """this function adds a task to the todoist api"""
    todoist.add_task(content=task, description=details)

tools = [add_task]
llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = gemini_api_key,
    temperature = 0.3
)

system_prompt = "you are a assistant. you will help the user add tasks"


prompt = ChatPromptTemplate([
    ("system", system_prompt),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),

])

#chain = prompt | llm | StrOutputParser()
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
#response = chain.invoke({"input": user_input})

history = []

while True:
    user_input = input("you:")
    response = agent_executor.invoke({"input": user_input, "history": history})
    print(response["output"])
    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=response["output"]))