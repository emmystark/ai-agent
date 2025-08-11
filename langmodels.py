# import getpass
# import os

# # Set Google API key securely
# if not os.environ.get("GOOGLE_API_KEY"):
#     os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# from langchain_google_genai import ChatGoogleGenerativeAI

# # Initialize the Gemini model
# model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# # Test the model with a simple prompt
# response = model.invoke("Hello! What is LangChain?")
# print(response.content)

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

# Load environment variables from .env file
load_dotenv()

# Verify Google API key is set
if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Define a simple tool for the agent
@tool
def sample_tool(query: str) -> str:
    """A sample tool that returns a greeting."""
    return f"Hello, {query}!"

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Test the model with a simple prompt
query = "Hi!"
response = model.invoke(query)
print("Model response:", response.content)

# Create a ReAct agent with the model and a tool
tools = [sample_tool]
agent_executor = create_react_agent(model, tools)

# Invoke the agent
input_message = {"role": "user", "content": "Hi!"}
response = agent_executor.invoke({"messages": [input_message]})

# Print the agent's response
for message in response["messages"]:
    message.pretty_print()
# print(response)