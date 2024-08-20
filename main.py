import os
import textwrap
import time
import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import Graph, END
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Environment setup
def setup_environment():
    os.environ["PINECONE_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    os.environ['GOOGLE_API_KEY'] = ""


# Pinecone initialization
def init_pinecone():
    return Pinecone(api_key=os.environ["PINECONE_API_KEY"])


# OpenAI and Google AI components initialization
def init_components():
    # embeddings = OpenAIEmbeddings(openai_api_base="https://apikey.dev.rapidinnovation.tech/")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_base="https://apikey.dev.rapidinnovation.tech/")
    return embeddings, llm


# CSV processing
def process_csv(csv_path):
    df = pd.read_csv(csv_path)
    df['combined_text'] = df['date_time'] + ' ' + df['name'] + ': ' + df['message']
    return [Document(page_content=text, metadata={"source": csv_path}) for text in df['combined_text']]


# Document splitting
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


# Pinecone index creation
def create_pinecone_index(pc, index_name):
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)


# Ingest documents to Pinecone
def ingest_to_pinecone(splits, embeddings, index_name):
    return PineconeVectorStore.from_documents(splits, embeddings, index_name=index_name)


# RAG Query Agent
def rag_query_agent(vectorstore, llm):
    def agent(state):
        query = state["query"]
        docs = vectorstore.similarity_search(query, k=3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the user's question."),
            ("human", "Context: {context}\n\nQuestion: {question}"),
        ])
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": query})
        state["initial_response"] = response
        return state

    return agent


# Reflection Agent
def reflection_agent(llm):
    def agent(state):
        query = state["query"]
        response = state["initial_response"]
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a reflective AI assistant. Analyze the given query and response, then provide insights on how to improve the answer."),
            ("human", "Query: {query}\n\nResponse: {response}\n\nReflection:"),
        ])
        chain = reflection_prompt | llm | StrOutputParser()
        reflection = chain.invoke({"query": query, "response": response})
        state["reflection"] = reflection
        return state

    return agent


# Final Answer Generation Agent
def final_answer_agent(llm):
    def agent(state):
        query = state["query"]
        response = state["initial_response"]
        reflection = state["reflection"]
        final_answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an AI assistant tasked with providing a comprehensive and improved answer. Use the original response and the reflection to create a better answer to the user's query."),
            (
            "human", "Query: {query}\n\nOriginal Response: {response}\n\nReflection: {reflection}\n\nImproved Answer:"),
        ])
        chain = final_answer_prompt | llm | StrOutputParser()
        final_answer = chain.invoke({"query": query, "response": response, "reflection": reflection})
        state["final_answer"] = final_answer
        return state

    return agent


# LangGraph setup
def setup_langgraph(rag_agent, reflection_agent, final_answer_agent):
    workflow = Graph()
    workflow.add_node("rag", rag_agent)
    workflow.add_node("reflection", reflection_agent)
    workflow.add_node("final_answer", final_answer_agent)

    workflow.set_entry_point("rag")
    workflow.add_edge("rag", "reflection")
    workflow.add_edge("reflection", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()


def format_output(text, width=80):
    """Format text to wrap at specified width."""
    return "\n".join(textwrap.wrap(text, width=width))


# Main execution function
def main():
    setup_environment()
    pc = init_pinecone()
    embeddings, llm = init_components()
    index_name = "meet"

    documents = process_csv("./meet.csv")
    splits = split_documents(documents)
    create_pinecone_index(pc, index_name)
    vectorstore = ingest_to_pinecone(splits, embeddings, index_name)

    rag_agent = rag_query_agent(vectorstore, llm)
    reflect_agent = reflection_agent(llm)
    final_agent = final_answer_agent(llm)

    chain = setup_langgraph(rag_agent, reflect_agent, final_agent)

    query = "What were the main topics discussed in the meeting?"
    result = chain.invoke({"query": query})

    print("\n" + "=" * 80)
    print("Reflective RAG Results".center(80))
    print("=" * 80)

    print("\nQuery:")
    print("-" * 80)
    print(format_output(query))

    print("\nInitial Response:")
    print("-" * 80)
    print(format_output(result["initial_response"]))

    print("\nReflection:")
    print("-" * 80)
    print(format_output(result["reflection"]))

    print("\nFinal Answer:")
    print("-" * 80)
    print(format_output(result["final_answer"]))

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
