# lags2.py — 로컬 RAG 분석 앱 (Ollama + Chroma + Gradio)

로컬 문서(`.txt`, `.pdf`, `.csv`, `.xlsx`, `.xls`)와 직접 수집한 웹 페이지를 벡터 DB(Chroma)에 넣고, 로컬 LLM(Ollama)으로 질문에 답하는 Gradio 웹 앱입니다. 최신판 `lags5.py`는 군사·전력 분석용 프롬프트를 사용합니다.

이 앱에서 버전마다 바꿔 온 설계(청크 크기, 임베딩 모델, 한→영 질의 확장)의 효과는 [AIN108/korean-rag-eval](https://github.com/AIN108/korean-rag-eval)에서 정량 비교합니다.

## 파일 구성

같은 앱의 버전별 사본입니다. **새로 실행할 때는 `lags5.py`를 사용하세요.**

| 파일 | 클래스 | 임베딩 모델 | 기본 LLM (Ollama) | 벡터 DB 폴더 |
|---|---|---|---|---|
| `rag.py` | `UltimateLocalRAG` | intfloat/multilingual-e5-large | llama3.1:70b | `vectorstore/chroma_db` |
| `lags2.py` | `UltimateLocalRAGv5` | intfloat/multilingual-e5-large | llama3.1:70b | `vectorstore/chroma_db_v5` |
| `lags3.py` | `BGEQwenRAGSystem` | BAAI/bge-m3 | qwen2.5:14b | `vectorstore/chroma_db_bge_qwen` |
| `lag4.py` | `UltimateMilitaryRAGv6` | intfloat/multilingual-e5-large | llama3.1:70b | `vectorstore/chroma_db_v6_military` |
| `lags5.py` | `UltimateMilitaryRAGv6` | intfloat/multilingual-e5-large | **qwen2.5:7b** (UI에서 gemma3n:e2b 선택 가능) | `vectorstore/chroma_db_v6_military` |

`lag4.py`와 `lags5.py`는 기본 모델과 선택 가능한 모델 목록만 다릅니다.

## 준비

1. **Ollama**를 설치하고 서버를 실행한 뒤 모델을 받습니다. 앱은 `http://localhost:11434`에 고정으로 접속합니다.
   ```bash
   ollama pull qwen2.5:7b
   ollama pull gemma3n:e2b   # 선택
   ```
2. **Python 가상환경**을 만들고 의존성을 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   - GPU를 쓰려면 `pip install -r requirements.txt` 전에 사용 중인 CUDA 버전에 맞는 PyTorch를 [pytorch.org](https://pytorch.org/get-started/locally/) 안내대로 먼저 설치하세요. GPU가 없으면 CPU로 동작합니다.
   - `requirements.txt`는 LangChain을 0.3 계열로 제한합니다. 현재 코드가 langchain 1.x에서 삭제된 `langchain.schema`를 import하기 때문입니다.
   - 첫 실행 때 Hugging Face에서 임베딩 모델(multilingual-e5-large, 약 2GB)을 내려받습니다.
3. `lags5.py`의 웹 수집은 requests + BeautifulSoup만 씁니다. 코드 상단에서 `playwright`, `trafilatura`, `newspaper`를 import하려고 시도하지만, `lags5.py`에서는 이 패키지들을 실제로 쓰지 않으므로 설치하지 않아도 됩니다. `rag.py`만 Playwright·trafilatura·newspaper3k를 사용합니다.

## 실행

```bash
python lags5.py
```

브라우저에서 `http://localhost:7860`이 열립니다(로컬 전용, `share=False`).

1. 시스템 초기화에서 모델(qwen2.5:7b / gemma3n:e2b)을 고릅니다.
2. 스크립트 옆에 자동으로 생기는 `research_data/` 폴더에 문서를 넣고 지식 베이스를 구축합니다. 새 파일만 추가하는 증분 업데이트도 지원합니다.
3. 분석 탭에서 질문합니다. 답변 끝에 참고한 파일·웹 출처가 표시됩니다.

`research_data/`, `vectorstore/`는 로컬 전용 데이터이며 `.gitignore`에 포함되어 있습니다.

## 알려진 제한

- Ollama 주소가 `http://localhost:11434`로 고정되어 있어 원격 Ollama 서버는 쓸 수 없습니다.
- 웹 수집 설정의 `respect_robots_txt` 값은 저장만 되고 실제로는 robots.txt를 확인하지 않습니다. 수집 대상 사이트의 이용 조건을 직접 확인하세요.
- 테스트 코드는 없습니다.
