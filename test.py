from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
import os
import tempfile
import streamlit as st


st.session_state.file_cache = {}

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

with st.sidebar:

    selected_model = st.selectbox(
            "사용할 LLM 모델 선택하기:",
            ("Phi-3", "Llama-3"),
            index=0,
            key='selected_model'  
        )

    st.header(f"PDF 추가하기")
    
    uploaded_file = st.file_uploader("`.pdf` 파일 업로드하기", type="pdf")

    if uploaded_file:
        try:
            file_key = f"{uploaded_file.name}"

            # 모델이 바뀌었는지와 캐시가 남아있는지 확인하기 
            if 'current_model' not in st.session_state or st.session_state.current_model != selected_model:
                st.session_state.current_model = selected_model
                st.session_state.file_cache.pop(file_key, None)  # 이전 모델 데이터 캐시 삭제하기
                st.experimental_rerun()  # 새로운 모델로 테스트하기 위해 재로딩하기

            # LLM 모델 셋업하기
            if st.session_state.current_model == "Llama-3":
                llm = Ollama(model="llama3", request_timeout=120.0)
            elif st.session_state.current_model == "Phi-3":
                llm = Ollama(model="phi3", request_timeout=120.0)

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            loader = SimpleDirectoryReader(
                                input_dir = temp_dir,
                                required_exts=[".pdf"],
                                recursive=True
                            )
                    else:    
                        st.error('다시 업로드해주세요...')
                        st.stop()
                    
                    docs = loader.load_data()

                    # 로딩된 데이터 인덱싱하기 
                    embed_model = Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

                    # 백터스토어에 업로드하기
                    index = VectorStoreIndex.from_documents(docs, show_progress=True)

                    Settings.llm = llm
                    # 검색 엔진 활성화하기 
                    query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)

                    # ====== Customise prompt template ======
                    qa_prompt_tmpl_str = (
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information above I want you to think step by step to answer the query in a crisp manner, incase case you don't know the answer say 'I don't know!'.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                    )
                    qa_prompt_tmpl = PromptTemplate(qa_prompt_tmpl_str)

                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_tmpl}
                    )
                    
                    st.session_state.file_cache[file_key] = query_engine
                else:
                    query_engine = st.session_state.file_cache[file_key]

                # 대화할 준비가 완료되었음을 알려주기
                st.success("Ready to Chat!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

col1, col2 = st.columns([6, 1])

with col1:
    st.header(f"Chat with your Docs! 📄")

with col2:
    st.button("Clear ↺", on_click=reset_chat)

# Initialize chat history
if "messages" not in st.session_state:
    reset_chat()


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Accept user input
if prompt := st.chat_input("What's up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate stream of response with milliseconds delay
        streaming_response = query_engine.query(prompt)
        
        for chunk in streaming_response.response_gen:
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})