import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_neo4j import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_experimental.graph_transformers import LLMGraphTransformer
import streamlit as st
import tempfile
from pyvis.network import Network
import networkx as nx

def visualize_graph(graph):
    """Fetch data from Neo4j and visualize it using pyvis."""
    G = nx.DiGraph()

    query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50"
    results = graph.query(query)

    for record in results:
        n = record["n"]
        m = record["m"]
        r = record["r"]

        n_label = n.get("name", n.get("id", "Node"))
        m_label = m.get("name", m.get("id", "Node"))

        G.add_node(n["id"], label=n_label, title=str(n))
        G.add_node(m["id"], label=m_label, title=str(m))
        G.add_edge(n["id"], m["id"], label=r)

    net = Network(notebook=False, height="600px", width="100%", directed=True)
    net.from_nx(G)
    
    # Generate HTML and display it in Streamlit
    net.write_html("graph.html")  # Write to an HTML file
    with open("graph.html", "r") as f:
        graph_html = f.read()

    st.write("### Graph Visualization")
    st.components.v1.html(graph_html, height=600, scrolling=True)

def main():
    st.set_page_config(layout="wide", page_title="KG Visualizer")
    with st.sidebar.expander("Expand Me"):
        st.markdown("""
        This application allows you to upload a PDF file, extract its content into a Neo4j graph database, and visualize it in real-time.
        """)

    st.title("Realtime Graph Visualization App")

    # Load environment variables
    load_dotenv()

    # OpenAI API Key setup from environment variable
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API key is not set in environment variables. Please configure it.")
        return

    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model_name="gpt-4")

    # Neo4j connection setup
    if 'neo4j_connected' not in st.session_state:
        st.sidebar.subheader("Connect to Neo4j Database")
        neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://<your-instance>.databases.neo4j.io")
        neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
        neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
        connect_button = st.sidebar.button("Connect")
        if connect_button and neo4j_password:
            try:
                graph = Neo4jGraph(
                    url=neo4j_url,
                    username=neo4j_username,
                    password=neo4j_password
                )
                st.session_state['graph'] = graph
                st.session_state['neo4j_connected'] = True
                st.sidebar.success("Connected to Neo4j database.")
            except Exception as e:
                st.error(f"Failed to connect to Neo4j: {e}")
    else:
        graph = st.session_state['graph']

    if 'graph' in st.session_state:
        uploaded_file = st.file_uploader("Please select a PDF file.", type="pdf")

        if uploaded_file is not None:
            with st.spinner("Processing the PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load_and_split()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
                docs = text_splitter.split_documents(pages)

                lc_docs = [
                    Document(page_content=doc.page_content.replace("\n", ""), metadata={'source': uploaded_file.name})
                    for doc in docs
                ]

                try:
                    graph.query("MATCH (n) DETACH DELETE n;")
                except Exception as e:
                    st.error(f"Failed to clear the graph database: {e}")

                transformer = LLMGraphTransformer(llm=llm)
                graph_documents = transformer.convert_to_graph_documents(lc_docs)
                graph.add_graph_documents(graph_documents, include_source=True)

                st.success(f"{uploaded_file.name} preparation is complete. The graph has been updated.")
                
                visualize_graph(graph)

if __name__ == "__main__":
    main()
