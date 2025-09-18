# gradio_chat_with_memory.py
import os
import gradio as gr

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory

# Your chat LLM client (you already had this working)
# Replace with whatever chat LLM wrapper you used (ChatGroq in your env)
from langchain_groq import ChatGroq

# -------------------------
# Config
# -------------------------
MODEL = "llama3-8b-8192"
GROQ_KEY = os.getenv("GROQ_API_KEY") or "YOUR_GROQ_API_KEY"

# -------------------------
# Initialize LLM + Memory + Prompt
# -------------------------
llm = ChatGroq(model=MODEL, api_key=GROQ_KEY)

# ConversationBufferMemory keeps chat history in this process
# return_messages=True makes load_memory_variables(...) return a list of message objects (preferred)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ChatPromptTemplate with a placeholder for chat history (inserted automatically)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and concise assistant."),
    ("placeholder", "{chat_history}"),      # <-- LangChain will insert history here
    ("human", "{user_input}")               # <-- the current user input
])

# -------------------------
# Helper to get history in the expected shape
# -------------------------
def _load_history_messages():
    raw = memory.load_memory_variables({}).get("chat_history", None)
    # Most modern LangChain returns a list of message objects when return_messages=True
    if not raw:
        return []
    # If your LangChain returns a string (older versions), fallback to empty list:
    if isinstance(raw, str):
        return []
    return raw  # assume list of message objects works with ChatPromptTemplate

# -------------------------
# Gradio respond function
# -------------------------
def respond(user_message, chat_history):
    """
    user_message: text input from the user (string)
    chat_history: gradio chat history list of tuples (user, bot)
    """
    # 1. load history messages from LangChain memory (list of message objects)
    hist_messages = _load_history_messages()

    # 2. format prompt messages (system + history + human)
    #    format_messages(...) returns a list of message objects (PromptValue/messages)
    prompt_messages = chat_prompt.format_messages(chat_history=hist_messages, user_input=user_message)

    # 3. call the LLM (pass the messages list / PromptValue)
    #    Many chat-llm wrappers accept the message list directly.
    #    If your wrapper expects a different shape, adapt here.
    response = llm.invoke(prompt_messages)

    # 4. extract assistant content (adjust if your response object differs)
    bot_text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)

    # 5. update LangChain memory (so next time history is available)
    try:
        # preferred API: add messages to chat_memory
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(bot_text)
    except Exception:
        # fallback API if methods are named differently
        try:
            memory.save_context({"input": user_message}, {"output": bot_text})
        except Exception:
            # last resort: ignore memory update (still works but no memory)
            pass

    # 6. update UI chat history (list of (user, bot) tuples for Gradio)
    chat_history = chat_history or []
    chat_history.append((user_message, bot_text))
    return chat_history, ""

# -------------------------
# Gradio UI
# -------------------------
with gr.Blocks() as demo:
    gr.Markdown("# Chatbot MVP — LangChain memory + ChatPromptTemplate + Gradio")
    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask me anything...", show_label=False)
    clear = gr.Button("Clear")

    # Submit user input
    user_input.submit(respond, [user_input, chatbot], [chatbot, user_input])
    # Clear resets the UI and also clears the LangChain memory
    def _clear_all():
        # reset LangChain memory storage for this session/demonstration
        try:
            # some versions expose memory.chat_memory.messages
            memory.chat_memory.messages = []
        except Exception:
            # fallback: recreate memory (simple)
            global memory
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        return [], ""
    clear.click(_clear_all, None, [chatbot, user_input])

demo.launch()
