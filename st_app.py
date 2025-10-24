import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import warnings

warnings.simplefilter("ignore", category=FutureWarning)
load_dotenv()

# Function to extract transcript and create vector store
@st.cache_resource(show_spinner=True)
def process_video(video_id):

    fetched_transcript = YouTubeTranscriptApi().fetch(video_id , languages=[ 'en'])
    full_text = " ".join([snippet.text for snippet in fetched_transcript])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([full_text])

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./models_hub"
    )

    vector_store = Chroma.from_documents(docs, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    return retriever

# Setup LLM
@st.cache_resource(show_spinner=False)
def load_llm():
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="conversational",
        max_new_tokens=512,
        temperature=0.5,
        top_k=50,
        repetition_penalty=1.03,
    )
    return ChatHuggingFace(llm=llm)

def build_chain(retriever, chat):
    prompt_template = """
    You are a helpful and honest assistant. Answer the question **strictly based** on the context provided below.  
    If the answer cannot be found in the context, respond with: **"I don't know"** or **"Not discussed in the transcript."**  
    Do not add any extra information or make assumptions beyond the given context.

    Context:
    {context}

    Question:
    {question} """
    prompt = PromptTemplate(template=prompt_template, input_variables={'context', 'question'})

    retrieval_chain = RunnableParallel({
        "context": retriever | RunnableLambda(lambda docs: "\n\n".join([doc.page_content for doc in docs])),
        "question": RunnablePassthrough(),
    })

    return retrieval_chain | prompt | chat | StrOutputParser()

# --- Streamlit App ---
st.set_page_config(page_title="YouTube RAG QA", layout="centered")
st.title("🎥 YouTube Video Q&A using RAG")

# Session state to hold retriever and chain
if "video_id" not in st.session_state:
    st.session_state.video_id = None
    st.session_state.retriever = None
    st.session_state.chain = None
    st.session_state.chat_model = load_llm()

# Step 1: Input YouTube Video ID
with st.sidebar:
    st.header("🔗 Video Setup")
    new_video_id = st.text_input("Enter YouTube Video ID:", value="", max_chars=20)

    if st.button("🔄 Load Video Transcript"):
        if new_video_id:
            with st.spinner("Processing video and creating embeddings..."):
                retriever = process_video(new_video_id)
                st.session_state.video_id = new_video_id
                st.session_state.retriever = retriever
                st.session_state.chain = build_chain(retriever, st.session_state.chat_model)
                st.success("Transcript and vector store ready!")
        else:
            st.warning("Please enter a valid video ID.")

    if st.button("🚪 Reset & Upload New Video"):
        st.session_state.clear()

# Step 2: Ask Questions
if st.session_state.retriever and st.session_state.chain:
    st.success(f"Video `{st.session_state.video_id}` loaded. You can now ask questions.")

    question = st.text_input("❓ Ask a question about the video")

    if question:
        with st.spinner("Getting answer..."):
            answer = st.session_state.chain.invoke(question)
            st.write("### 📥 Answer:")
            st.write(answer)

else:
    st.info("Please enter a YouTube video ID in the sidebar to begin.")
