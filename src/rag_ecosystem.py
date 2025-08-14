import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()

###### 1 ################ INDEXING  ###################################################################

#### Step 1 of Indexing : Reading the source 
# Initialize a web document loader with specific parsing instructions
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),  # URL of the blog post to load
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")  # Only parse specified HTML classes
        )
    ),
)

# Load the filtered content from the web page into documents
docs = loader.load()


#### Step 2 of Indexing : Text Splitting
# Create a text splitter to divide text into chunks of 1000 characters with 200-character overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Split the loaded documents into smaller chunks
splits = text_splitter.split_documents(docs)



#### Step 3 of Indexing : Vectorization = Converting the chunks of texts into numerical representations called embeddings
# Then store these embeddings in a Vector database

# Embed the text chunks and store them in a Chroma vector store for similarity search
vectorstore = Chroma.from_documents(
    documents=splits, 
    embedding=OpenAIEmbeddings()  # Use OpenAI's embedding model to convert text into vectors
)

###### 2 ################ RETRIEVAL  ###################################################################

# Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# Retrieve relevant documents for a query
docs = retriever.get_relevant_documents("What is Task Decomposition?")

# Print the content of the first retrieved document
print(docs[0].page_content)

# As you can see, the retriever successfully pulled the most relevant chunk from the 
# blog post that directly discusses “Task decomposition.” 
# This piece of context is exactly what the LLM needs to form an accurate answer.





###### 3 ################ GENERATION  ###################################################################

# Pull a pre-made RAG prompt from LangChain Hub
prompt = hub.pull("rlm/rag-prompt")

# printing the prompt
print(prompt)

# Initialize the LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the full RAG chain

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

""" 
Let’s break down this chain:
{"context": retriever | format_docs, "question": RunnablePassthrough()}: This part runs in parallel. It sends the user's question to the retriever to get documents, which are then formatted into a single string by format_docs. Simultaneously, RunnablePassthrough passes the original question through unchanged.
| prompt: The context and question are fed into our prompt template.
| llm: The formatted prompt is sent to the LLM.
| StrOutputParser(): This cleans up the LLM's output into a simple string.
#
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
) """

# Ask a question using the RAG chain
response = rag_chain.invoke("What is Task Decomposition?")
print(response)

""" And there we have it, our RAG pipeline successfully retrieved relevant information about “Task Decomposition” 
and used it to generate a concise, accurate answer. 
This simple chain forms the foundation upon which we will build more advanced and powerful capabilities """