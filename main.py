import os

from dotenv import load_dotenv
import os
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from openai import OpenAI
from langchain_tavily import TavilySearch

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is missing from .env or environment")

# ... then pass it to clients
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY)

# completion = client.chat.completions.create(
#   model="meta/llama-3.3-70b-instruct",
#   messages=[{"role":"user","content":""}],
#   temperature=0.2,
#   top_p=0.7,
#   max_tokens=1024,
#   stream=False
# )

# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",api_key=NVIDIA_API_KEY)
tools = [TavilySearch()]
agent = create_agent(tools=tools, model=llm)


def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="weather in salem.india now")})
    print(result)


if __name__ == "__main__":
    main()
