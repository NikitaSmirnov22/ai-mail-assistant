import os
import getpass
from flask import Request
import streamlit as st
from typing import List, TypedDict
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from langchain.chat_models import ChatOpenAI  # Updated import
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.agent_toolkits import GmailToolkit
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain import hub

# Prompt for the OpenAI API key securely and set it as an environment variable
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API Key: ")

# Path to your credentials.json file
CLIENT_SECRETS_FILE = 'credentials.json'

# Define all necessary Google API scopes
SCOPES = [
    # Gmail
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.insert",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.labels",
    "https://www.googleapis.com/auth/gmail.metadata",
    "https://www.googleapis.com/auth/gmail.settings.basic",
    "https://www.googleapis.com/auth/gmail.settings.sharing",

    # Google Drive
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.metadata",
    "https://www.googleapis.com/auth/drive.metadata.readonly",
    "https://www.googleapis.com/auth/drive.appdata",
    "https://www.googleapis.com/auth/drive.photos.readonly",

    # Google Calendar
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/calendar.events.readonly",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.settings.readonly",

    # Contacts and People API
    "https://www.googleapis.com/auth/contacts",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/contacts.other.readonly",
    "https://www.googleapis.com/auth/directory.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/user.phonenumbers.read",
    "https://www.googleapis.com/auth/user.addresses.read",

    # YouTube
    "https://www.googleapis.com/auth/youtube",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtubepartner"
]

# Function to get or refresh Google credentials
def get_google_credentials():
    creds = None
    # Load existing credentials if available
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    # Authenticate or refresh if credentials are invalid
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRETS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)  # Automatically finds an available port
        # Save the new credentials for future use
        with open("token.json", 'w') as token_file:
            token_file.write(creds.to_json())
    return creds

# Authenticate and build Gmail API service
try:
    credentials = get_google_credentials()
    api_resource = build("gmail", "v1", credentials=credentials)
except Exception as e:
    st.write("Error building Gmail API resource:", e)

# Initialize the Gmail Toolkit
toolkit = GmailToolkit(api_resource=api_resource)
tools = toolkit.get_tools()

# Initialize the LLM with the required API key
llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=os.environ["OPENAI_API_KEY"])

# Create the agent using OpenAI Functions Agent
instructions = "You are an assistant that helps manage emails and performs web searches."
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_openai_functions_agent(llm, tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,
)

# Define State
class State(TypedDict):
    messages: List[BaseMessage]
    ask_human: bool

# Define the chatbot function
def chatbot(state: State):
    user_message = state["messages"][-1]
    agent_input = {"input": user_message.content}
    try:
        response = agent_executor.invoke(agent_input)
        ai_content = response['output']
    except Exception as e:
        ai_content = f"An error occurred: {e}"
    ai_message = AIMessage(content=ai_content)
    state["messages"].append(ai_message)
    ask_human = "unable to fulfill" in ai_content.lower()
    return {"messages": state["messages"], "ask_human": ask_human}

# Streamlit Interface
st.title("AI-Driven Email and Search Assistant")

if 'state' not in st.session_state:
    st.session_state['state'] = {"messages": [], "ask_human": False}

query = st.text_input("Enter your search query or email request")

if st.button("Execute"):
    state = st.session_state['state']
    state["messages"].append(HumanMessage(content=query))
    response = chatbot(state)
    if response["ask_human"]:
        st.write("### Request for Human Assistance")
        st.write("Unable to fulfill the request autonomously.")
    else:
        st.write("### Response")
        st.write(response["messages"][-1].content)
