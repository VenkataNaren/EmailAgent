"""
A simple agent that runs on your local Ollama model.

Setup:
    pip install ollama
Run:
    python simple_agent.py
"""

import datetime
import os

import ollama
import truststore
truststore.inject_into_ssl()

MODEL = "llama3.1:8b"
MAX_STEPS = 5  # safety limit so the agent can't loop forever


# ---- The agent's "hands": plain Python functions ----------------------------
# The docstrings matter. The model reads them to decide when to use each tool.

def get_current_time() -> str:
    """Get the current date and time on this computer."""
    return datetime.datetime.now().strftime("%A %d %B %Y, %I:%M %p")


def calculate(expression: str) -> str:
    """Calculate a math expression such as '23 * 7 + 4'. Use this for any arithmetic."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only numbers and + - * / ( ) are allowed."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def list_files(folder: str = ".") -> str:
    """List the files and folders inside a folder on this computer."""
    try:
        return ", ".join(os.listdir(folder)[:50])
    except Exception as e:
        return f"Error: {e}"


TOOLS = {f.__name__: f for f in (get_current_time, calculate, list_files)}

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use a tool only when you need it "
    "(time, math, or listing files). Otherwise just answer directly. "
    "Keep answers short."
)


# ---- The agent loop ----------------------------------------------------------

def run_agent(messages: list) -> str:
    for _ in range(MAX_STEPS):
        response = ollama.chat(model=MODEL, messages=messages, tools=list(TOOLS.values()))
        messages.append(response.message)

        # No tool requested -> the model has its final answer.
        if not response.message.tool_calls:
            return response.message.content

        # Otherwise run each requested tool and give the result back to the model.
        for call in response.message.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"  [using tool: {name}({args})]")
            func = TOOLS.get(name)
            result = func(**args) if func else f"Unknown tool: {name}"
            messages.append({"role": "tool", "tool_name": name, "content": str(result)})

    return "Sorry, I got stuck. Try asking in a different way."


def main():
    print(f"Agent ready (model: {MODEL}). Type 'exit' to quit.\n")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        print("Agent:", run_agent(messages), "\n")


if __name__ == "__main__":
    main()
