import os
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load API Key
load_dotenv()

st.set_page_config(page_title="Institution AI", page_icon="🏫")
st.title("🏫 Student Assistant Bot")

# 1. Setup LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Setup Vector Database (For Question Papers & Syllabus)
@st.cache_resource
def setup_knowledge_base():
    # In production, replace this with your OCR pipeline for scanned papers
    if not os.path.exists("./data"):
        os.makedirs("./data")
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = setup_knowledge_base()

# 3. Create Tools for the Agent
tools = []

# Tool A: PDF Search
if retriever:
    pdf_tool = create_retriever_tool(
        retriever,
        "search_institutional_records",
        "Searches and returns information regarding syllabus, past question papers, and campus rules."
    )
    tools.append(pdf_tool)

# Tool B: Timetable Database Query (Simulated)
@tool
def get_student_timetable(student_id: str, day: str) -> str:
    """Fetches the class timetable for a specific student on a given day."""
    # In production, this connects to your SQL/NoSQL database via SQLAlchemy/PyMongo
    timetable_db = {
        "Monday": "10:00 AM - Data Structures, 11:30 AM - Database Management Systems",
        "Tuesday": "09:00 AM - Operating Systems, 01:00 PM - Software Engineering Lab"
    }
    schedule = timetable_db.get(day.capitalize(), "No classes scheduled or invalid day.")
    return f"Timetable for {student_id} on {day}: {schedule}"

tools.append(get_student_timetable)

# 4. Initialize the Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful university assistant. Use your tools to answer student queries. "
               "If asked about schedules, always ask for the student ID and day if not provided."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. Streamlit UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask about syllabus, or check your timetable..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = agent_executor.invoke({"input": user_input})
            answer = response["output"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})