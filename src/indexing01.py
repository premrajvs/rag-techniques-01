# reading web page
import bs4
# loading document into documents
from langchain_community.document_loaders import WebBaseLoader

# chunking
from langchain.text_splitter import RecursiveCharacterTextSplitter

# embedding
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class indexingoutput:
    def __init__(self, vectorstore=None, docs=None):
        self.vectorstore = vectorstore
        self.docs = docs

class indexing01:
    def __init__(self):
        pass

    def runindexing(self):
        docs = self.indexing()
        splits = self.textchunking(docs)
        vectorstore = self.createembeddings(splits)

        return vectorstore#indexingoutput(vectorstore=vectorstore, docs=docs)

    def indexing(self):
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
        return docs

    def textchunking(self, docs):
        # Create a text splitter to divide text into chunks of 1000 characters with 200-character overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Split the loaded documents into smaller chunks
        splits = text_splitter.split_documents(docs)
        return splits
    
    def createembeddings(self,splits):
        # Embed the text chunks and store them in a Chroma vector store for similarity search
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=OpenAIEmbeddings()  # Use OpenAI's embedding model to convert text into vectors
        )
        return vectorstore