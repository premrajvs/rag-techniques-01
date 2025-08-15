class retrieval01:
    def __init__(self, vectorstore=None, question=None):
        self.vectorstore = vectorstore
        self.question = question

    def runretrieval(self):
        """Runs the retrieval process."""
        retriever, docs = self.retrieval()
        # Print the content of the first retrieved document for debugging
        if docs:
            print("--- Top Retrieved Document ---")
            print(docs[0].page_content)
            print("------------------------------\n")
        return retriever, docs

    def retrieval(self):
        """Creates a retriever and gets relevant documents."""
        retriever = self.vectorstore.as_retriever()
        # Retrieve relevant documents for a query
        docs = retriever.get_relevant_documents(self.question)
        return retriever, docs