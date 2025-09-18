from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import gradio as gr
import sys
import types

# prevent import error for audioop
sys.modules['pyaudioop'] = types.ModuleType('pyaudioop')

# 🔑 Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="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",  
    max_tokens=512,
    temperature=0.2
)

# 🧠 Memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# 🔮 Prompt Template (with RAG context + memory)
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant named Jarvis for design engineers.
Use the following retrieved context to answer the question:

Context:
{context}

Conversation so far:
{chat_history}

User Question: {user_input}

Answer in detail:
""")

# Vector store (global, will be updated when PDFs are uploaded)
vectorstore = None

def process_pdf(pdf_file):
    global vectorstore
    loader = PyPDFLoader(pdf_file.name)
    documents = loader.load()

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    # Create vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    return "✅ PDF uploaded and processed. You can now start asking questions!"

# Load history helper
def _load_history_messages():
    raw = memory.load_memory_variables({}).get("chat_history", None)
    if not raw:
        return []
    if isinstance(raw, str):
        return []
    return raw

# Respond function with retrieval
def respond(user_message, chat_history):

    global vectorstore

    if vectorstore is None:
        return [("System", "⚠️ Please upload a PDF first before asking questions.")], ""

    # hist_messages = _load_history_messages()

      # Retrieve relevant context from PDF
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(user_message)
    context = "\n".join([doc.page_content for doc in docs])

    # Format prompt
    hist_messages = memory.load_memory_variables({}).get("chat_history", [])
    prompt_messages = prompt.format_messages(
    chat_history=hist_messages,
    user_input=user_message,
    context=context  
)

    # Get LLM response
    response = llm.invoke(prompt_messages)
    bot_text = getattr(response, "content", None) or getattr(response, "text", None) or str(response)

    # Save to memory
    try:
        memory.chat_memory.add_user_message(user_message)
        memory.chat_memory.add_ai_message(bot_text)
    except Exception:
        try:
            memory.save_context({"input": user_message}, {"output": bot_text})
        except Exception:
            pass

    chat_history = chat_history or []
    chat_history.append((user_message, bot_text))
    return chat_history, ""

# 🧹 Clear chat
def _clear_all():
    global memory
    try:
        memory.chat_memory.messages = []
    except Exception:
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return [], ""

# 🎨 UI with Gradio
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1 style='text-align:center; color:#4B0082;'>🤖 Jarvis - Design Engineer Assistant</h1>
        <p style='text-align:center; font-size:16px; color:#333;'>Ask me questions from the Saudi Aramco SAMS specification.</p>
        """, 
        elem_id="title"
    )
    with gr.Row():
        pdf_upload = gr.File(label="Upload PDF", file_types=[".pdf"])
        upload_output = gr.Textbox(label="Upload Status", interactive=False)

    chatbot = gr.Chatbot()
    user_input = gr.Textbox(placeholder="Ask about the SAMS specification...", show_label=False)
    clear = gr.Button("Clear")

    pdf_upload.upload(process_pdf, pdf_upload, upload_output)
    user_input.submit(respond, [user_input, chatbot], [chatbot, user_input])
    clear.click(_clear_all, None, [chatbot, user_input])

demo.launch(share=True)
