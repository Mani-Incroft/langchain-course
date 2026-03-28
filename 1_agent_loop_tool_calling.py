from dotenv import load_dotenv
import os
load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langsmith import traceable
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from openai import OpenAI

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
MODEL = os.getenv("NVIDIA_MODEL")

if not NVIDIA_API_KEY:
    raise RuntimeError("NVIDIA_API_KEY is missing from .env or environment")

# ... then pass it to clients
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY)

MAX_ITERATIONS = 10
# MODEL="gpt-oss:20b"

# Tools (Langchain @tool decorator)

@tool 
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f". >> Executing get_product_price(product_id='{product}')")
    prices = {"laptop": 1299.99, "headphones": 19.99, "mouse": 29.99}
    return prices.get(product,0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price and return the final price. Available tiers: bronze, silver, gold."""
    print(f". >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount/100),2)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
   tools = [get_product_price, apply_discount]
   tool_dict = {t.name: t for t in tools}

#    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
   llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct",api_key=NVIDIA_API_KEY,temperature=0)
   llm_with_tools = llm.bind_tools(tools)

   print(f"Question: {question}")
   print("="*70)

   messages = [
       SystemMessage(
        content=("You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price function FIRST with product nameto get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
       )
       ),
       HumanMessage(content=question),
   ]

   for iteration in range(1, MAX_ITERATIONS+1):
       print(f"\nIteration: {iteration} ---")

       ai_message = llm_with_tools.invoke(messages)

       tool_calls = ai_message.tool_calls

       if not tool_calls:
           print(f"\n Final answer: {ai_message.content}")
           return ai_message.content

       # Process tool calls
       tool_call = tool_calls[0]  # Assuming one tool call per iteration for simplicity
       tool_name = tool_call.get("name")
       tool_args = tool_call.get("args", {})
       tool_call_id = tool_call.get("id")

       print(f". [Tool Selected] {tool_name} with args: {tool_args}")
       
       tool_to_use = tool_dict.get(tool_name)
       if tool_to_use is None:
           raise ValueError(f"Tool {tool_name} not found")
       
       observation = tool_to_use.invoke(tool_args)
       print(f" [Tool Results] {observation}")
           
        #    # Append tool result to messages
       messages.append(ai_message)
       messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
       )
   print("Error: Max iterations reached without final answer.")
   return None
       

if __name__ == "__main__":
    result = run_agent("What is the price of a laptop after applying a silver discount?")
    print(f"Final Result: {result}")
