import indexing01
import retrieval01
import generateoutput01
from dotenv import load_dotenv

class ragimplementation:
    def __init__(self, indexing_output=None, retrieval_output=None, question=None):
        self.indexing_output = indexing_output
        self.retrieval_output = retrieval_output
        self.question = question

    def run_rag(self):
        indexing_instance = indexing01.indexing01()
        self.indexing_output = indexing_instance.runindexing()

        self.question = "What is Task decomposition?"
        retrieval_instance = retrieval01.retrieval01(vectorstore=self.indexing_output, question=self.question)

        self.retrieval_output = retrieval_instance.runretrieval()

        # The retriever object is the first element of the tuple returned by runretrieval
        retriever = self.retrieval_output[0]
        generate_instance = generateoutput01.generateoutput(ragretriever=retriever)
        generate_instance.run_generate()

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()
    rag = ragimplementation()
    rag.run_rag()