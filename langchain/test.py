

qa = RetrievalQA.from_chain_type(llm = llm,
                                 chain_type = "stuff",
                                 retriever = docsearch.as_retriever(
                                    search_type="mmr",
                                    search_kwargs={'k':3, 'fetch_k': 10}),
                                 return_source_documents = True)

query = "체크리스트로 만들어줘"
result = qa(query)