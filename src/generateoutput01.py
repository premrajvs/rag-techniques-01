# for prompt
from urllib import response
from langchain import hub

# for LLM
from langchain_openai import ChatOpenAI

# for output parsing
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class generateoutput:
    def __init__(self, ragretriever=None):
        self.ragretriever = ragretriever

    def run_generate(self):
        prompt = self.prompt_setup()
        llm = self.llm_setup()

        self.run_rag_chain(self.ragretriever, prompt, llm)

    def prompt_setup(self):
        # Pull a pre-made RAG prompt from LangChain Hub
        prompt = hub.pull("rlm/rag-prompt")
        
        print("--- RAG Prompt Template ---")
        # printing the prompt
        print(prompt)
        return prompt

    def llm_setup(self):
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        return llm

    # Helper function to format retrieved documents
    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def run_rag_chain(self, retriever,prompt,llm):
        # Define the full RAG chain
        rag_chain = (
            {"context": retriever | generateoutput.format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        print("\n--- Generated Answer ---")
        response = rag_chain.invoke("What is Task Decomposition?")
        print(response)