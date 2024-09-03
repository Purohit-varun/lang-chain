from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document

load_dotenv()


class RAG:
    def __init__(self) -> None:
        self.memory_file = "memory.json"
        self.embedding_function = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ".",
                ",",
                "",
            ],
        )
        self.documents = self.load_and_split("C:/Varun/JS/Rag-chatbot/lang-chain/rag-chatbot/facts.txt")

        self.vectorstore = Chroma.from_documents(self.documents, self.embedding_function)

        self.prompt_template = """You are an AI Assistant having a conversation with a human.

            You will be given the extracted parts of a long document as a context, and a question related to that context.

            Use only and only the provided context to answer the question. DO NOT use your own knowledge base to answer the questions.

            ### Context:
            {context}

            ### Query:
            Human: {human_input}
            AI:"""
        
        self.base_prompt = PromptTemplate(
            input_variables=["human_input", "context"],
            template=self.prompt_template,
        )

        self.chain = load_qa_chain(
            llm=self.llm, chain_type="stuff", prompt=self.base_prompt
        )


    
    def load_and_split(self, file_path: str):
        # Step 1: Load the document
        loader = TextLoader(file_path)
        documents = loader.load()
        
        # Step 2: Split the document into chunks
        texts = [doc.page_content for doc in documents]
        split_docs = self.text_splitter.split_text(" ".join(texts))
        
        # Create Document objects from the split text
        return [Document(page_content=chunk) for chunk in split_docs]

    def ask_question(self, question: str):
        # Search for relevant documents using the vectorstore
        docs = self.vectorstore.similarity_search(question)
        
        # Extract the page content from the relevant documents
        context = " ".join([doc.page_content for doc in docs])
        
        # Generate a response using the QA chain
        response = self.chain.run({"human_input": question, "context": context})
        return response


# Usage Example:
if __name__ == "__main__":
    rag = RAG()
    question = "What is the significance of the document?"
    answer = rag.ask_question(question)
    print(answer)
 
    # def load_conversation(self, file_path):
    #     with open(file_path, 'r') as file:
    #         conversation = json.load(file)
            
    #     # Get the latest 10 messages
    #     latest_messages = conversation[-10:]

    #     formatted_output = []
    #     for message in latest_messages:
    #         message_type = message['type']
    #         content = message['data']['content']
    #         formatted_output.append(f"{message_type}: {content}")
        
    #     return "\n".join(formatted_output)

    # # def load_db(self, file_path: str, document_id:str):
    # def load_db(self, file_path: str):
    #     # success = self.ingest_documents(file_path=file_path, document_id=document_id)
    #     success = self.ingest_documents(file_path=file_path)
    #     return success

    # # def ingest_documents(self, file_path:str, document_id:str):
    # def ingest_documents(self, file_path:str):
    #     if ".pdf" in file_path:
            
    #     elif ".md" in file_path:
    #         self.loader = UnstructuredMarkdownLoader(file_path)
    #     elif ".txt" in file_path:
            
    #     elif ".docx" in file_path:
    #         self.loader = Docx2txtLoader(file_path)
    #     elif ".pptx" in file_path:
    #         self.loader = UnstructuredPowerPointLoader(file_path=file_path)
    #     self.document_list = self.loader.load_and_split(
    #         text_splitter=self.text_splitter
    #     )
    #     for document in self.document_list:
    #         document.metadata = {"user_id": self.user_id}
    #         # document.metadata = {"document_id": document_id}

    #     # Connect to Zilliz Cloud and create the vector store
    #     self.vector_store = Milvus.from_documents(
    #         documents=self.document_list,
    #         embedding=self.embedding_function,
    #         connection_args=self.connection_args
    #     )
    #     return True

    def invoke(self, query: str):
        self.vector_store = Milvus(
            embedding_function=self.embedding_function,
            connection_args=self.connection_args,
        )
        retriever = self.vector_store.as_retriever(
            search_type="similarity"
        )
        retrieved_docs = retriever.invoke(query)
        conversation = self.load_conversation(self.memory_file)
        self.response = self.chain.invoke(
            {"input_documents": retrieved_docs, "human_input": query, "chat_history":conversation},
            return_only_outputs=True
        )
        self.chat_memory.add_user_message(query)
        self.chat_memory.add_ai_message(self.response["output_text"])
        return self.response["output_text"]


