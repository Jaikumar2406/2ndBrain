from flask import Flask, render_template, request, jsonify, session
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
import os
from dotenv import load_dotenv
import uuid

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')

# Initialize MongoDB connection
client = MongoClient(os.getenv('client_env'))
db = client["2ndBrain"]
task_collection = db["tasks"]
user_collection = db["users"]
user_calendarevent = db["calenderevent"]

# Initialize LLM
model = ChatGroq(api_key=os.getenv('groq_api_key'), model="llama-3.3-70b-versatile", temperature=0.5)

# Embeddings
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"token": os.getenv('hf_token')}
)

from langchain_core.documents import Document


def get_tasks_as_documents():
    tasks = task_collection.find({})
    return [
        Document(
            page_content=task["description"],
            metadata={
                "task_id": str(task["_id"]),
                "user_id": str(task.get("user_id", "")),
                "due_date": task.get("due_date", ""),
                "status": task.get("status", "pending")
            }
        ) for task in tasks
    ]


docs = get_tasks_as_documents()

# Initialize Vector Store
vector_store = Chroma.from_documents(docs, embedding)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Contextualize question prompt
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    model,
    retriever,
    contextualize_q_prompt
)

# QA Prompt Template
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """As a smart task scheduler, your role is to help users manage their tasks efficiently.  

- If the user provides multiple tasks, analyze their importance and urgency, then rearrange them into a structured timetable.  
- If the user gives a single task, break it into smaller steps for better execution.  
- If a task is marked as completed, remove it from the schedule and display only the remaining tasks.  
- If the user asks something unrelated to task scheduling, politely inform them that you can only assist with task management.
- If the user asks about today's work, check the database and arrange tasks into a structured timetable.

Provide a well-balanced schedule to prevent overload and optimize productivity.  

CONTEXT: {context}  

QUESTION: {input}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# Create chains
question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Chat history store
store = {}


# Function to get or create session history
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Creating a chain with memory
chain_with_memory = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


@app.route('/')
def home():
    # Generate a unique session ID for each user if it doesn't exist
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    session_id = session.get('session_id', str(uuid.uuid4()))

    # Get chatbot response
    response = chain_with_memory.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )

    return jsonify({
        'response': response['answer']
    })


if __name__ == '__main__':
    app.run(debug=True)