from dotenv import load_dotenv
import getpass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


os.environ["OPENAI_API_KEY"] = getpass.getpass()



class RAG:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)
