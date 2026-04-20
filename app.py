import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# Load API Key
load_dotenv()

st.set_page_config(page_title="Institution AI", page_icon="🏫")
st.title("🏫 Student Assistant Bot")

# 1. Setup LLM (Using the fast, stable Flash model)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Setup Vector Database (Strictly for PDFs like Syllabus)
@st.cache_resource
def setup_pdf_knowledge_base():
    if not os.path.exists("./data"):
        os.makedirs("./data")
    loader = PyPDFDirectoryLoader("./data")
    docs = loader.load()
    if not docs:
        return None
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = setup_pdf_knowledge_base()

# 3. Create Agent Tools
tools = []

# Tool A: PDF Searcher (Syllabus/Rules)
if retriever:
    pdf_tool = create_retriever_tool(
        retriever,
        "search_syllabus_and_rules",
        "Use this tool to answer questions about the course syllabus, campus rules, or general institutional guidelines."
    )
    tools.append(pdf_tool)

# Tool B: Timetable Searcher (Reads CSV)
@tool
def check_timetable(query_type: str, day_or_date: str) -> str:
    """
    Fetches timetable data. 
    query_type must be either 'regular' or 'end_sem'.
    day_or_date is the day of the week (e.g., 'Monday') or exam date.
    """
    try:
        if query_type == 'regular':
            df = pd.read_csv("./data/regular_timetable.csv")
            # Filter rows matching the requested day (case-insensitive)
            result = df[df['Day'].str.contains(day_or_date, case=False, na=False)]
        else:
            df = pd.read_csv("./data/end_sem_timetable.csv")
            result = df[df['Date'].str.contains(day_or_date, case=False, na=False)]
            
        if result.empty:
            return f"No schedule found for {day_or_date}."
        return result.to_string(index=False) # Returns the table data as a clean string for the LLM
    except Exception as e:
        return "Error reading timetable data. Please ensure the CSV files exist in the data folder."

tools.append(check_timetable)

# Tool C: Results Searcher (Reads CSV)
@tool
def check_student_results(student_id: str) -> str:
    """Fetches the exam results and grades for a specific student ID."""
    try:
        df = pd.read_csv("./data/results.csv")
        # Ensure student_id is treated as a string for accurate matching
        df['Student_ID'] = df['Student_ID'].astype(str)
        result = df[df['Student_ID'] == str(student_id)]
        
        if result.empty:
            return f"No results found for Student ID: {student_id}."
        return result.to_string(index=False)
    except Exception as e:
        return "Error reading results database."

tools.append(check_student_results)

# 4. Initialize the Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful university assistant. You have tools to check the syllabus, timetables, and student results. "
               "If a student asks for their result or schedule, ask for their Student ID or the specific day if they didn't provide it."),
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

if user_input := st.chat_input("Ask about syllabus, timetables, or results..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching institutional records..."):
            try:
                response = agent_executor.invoke({"input": user_input})
                answer = response["output"]
            except Exception as e:
                answer = f"I encountered an error while searching: {str(e)}"
                
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
