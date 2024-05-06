## 실행 방법

```
python -m venv venv
source venv/bin/activate
pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama
pip install streamlit
streamlit run test.py

```

## 여러 LLM 모델 사용하는 방법

터미널에서 Ollama를 사용하여 여러 LLM 모델들을 먼저 다운받아야 합니다.

```
ollama pull llama3
ollama pull phi3

```
