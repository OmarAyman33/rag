# Current RAG design & Future Steps

## Current Design

First we split our document database (currently just scraped wikipedia articles) into chunks using LangChain's `RecursiveCharacterTextSplitter` (which just depends on document structure).

After that, we generate embeddings for the each chunk using google's `embedding gemma 300m` model. These chunks and their respective embeddings are stored with the help of chromaDB as a vector database.

Then when the user enters their prompt, the prompt gets turned into embeddings, and then the top K (currently 5) relevant chunks in cosine similarity are retreived and injected into the prompt. Which then the model uses in its answer.

## Future Steps & Design Edits

1. Add a reranker so that the RAG system does not just depend on the embedding model's output and has a second layer of checking of semantic correlation of the chunks with the prompt. 
2. Instead of feeding the prompt directly into the RAG system / the embedding model. First enter the prompt into the LLM with a prompt that asks it to split the prompt into simple atomic questions. these questions are then what is used for retreival and therefore chunks will be more relevant to individual pieces of information. 
3. add gaurdrails of some form. (mkasel akteb)