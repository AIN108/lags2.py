# lags2.py — 로컬 RAG 분석 앱 (Ollama + Chroma + Gradio)

로컬 문서(`.txt`, `.pdf`, `.csv`, `.xlsx`, `.xls`)와 직접 수집한 웹 페이지를 벡터 DB(Chroma)에 넣고, 로컬 LLM(Ollama)으로 질문에 답하는 Gradio 웹 앱입니다. 최신판 `lags5.py`는 군사·전력 분석용 프롬프트를 사용합니다.

이 앱에서 버전마다 바꿔 온 설계(청크 크기, 임베딩 모델, 한→영 질의 확장)의 효과는 [AIN108/korean-rag-eval](https://github.com/AIN108/korean-rag-eval)에서 정량 비교합니다.

## 파일 구성

최신판은 저장소 루트의 `lags5.py`입니다. **새로 실행할 때는 `lags5.py`를 사용하세요.** 이전 버전 사본은 기록용으로 `legacy/` 폴더에 보관합니다.

| 파일 | 클래스 | 임베딩 모델 | 기본 LLM (Ollama) | 벡터 DB 폴더 |
|---|---|---|---|---|
| `legacy/rag.py` | `UltimateLocalRAG` | intfloat/multilingual-e5-large | llama3.1:70b | `legacy/vectorstore/chroma_db` |
| `legacy/lags2.py` | `UltimateLocalRAGv5` | intfloat/multilingual-e5-large | llama3.1:70b | `legacy/vectorstore/chroma_db_v5` |
| `legacy/lags3.py` | `BGEQwenRAGSystem` | BAAI/bge-m3 | qwen2.5:14b | `legacy/vectorstore/chroma_db_bge_qwen` |
| `legacy/lag4.py` | `UltimateMilitaryRAGv6` | intfloat/multilingual-e5-large | llama3.1:70b | `legacy/vectorstore/chroma_db_v6_military` |
| `lags5.py` | `UltimateMilitaryRAGv6` | intfloat/multilingual-e5-large | **qwen2.5:7b** (UI에서 gemma3n:e2b 선택 가능) | `vectorstore/chroma_db_v6_military` |

`legacy/lag4.py`와 `lags5.py`는 기본 모델과 선택 가능한 모델 목록만 다릅니다. 옛 사본은 스크립트가 있는 폴더 기준으로 데이터를 찾으므로 `legacy/research_data/`, `legacy/vectorstore/`를 씁니다. 예전에 루트에서 옛 사본을 실행하며 만든 `research_data/`·`vectorstore/`를 계속 쓰려면 `legacy/` 아래로 옮기세요.

## 준비

1. **Ollama**를 설치하고 서버를 실행한 뒤 모델을 받습니다. 앱은 기본으로 `http://localhost:11434`에 접속합니다.
   ```bash
   ollama pull qwen2.5:7b
   ollama pull gemma3n:e2b   # 선택
   ```
   다른 PC나 다른 포트의 Ollama를 쓰려면 Ollama CLI와 같은 `OLLAMA_HOST` 환경변수를 지정합니다. 스킴 없이 호스트만 쓰면 `http://`와 기본 포트 `11434`가 붙습니다.
   ```bash
   OLLAMA_HOST=192.168.0.10 python lags5.py              # → http://192.168.0.10:11434
   OLLAMA_HOST=http://gpu-box:8080 python lags5.py       # 스킴을 쓰면 그대로 사용
   ```
   Windows PowerShell에서는 `$env:OLLAMA_HOST="192.168.0.10"`로 지정한 뒤 실행합니다.
2. **Python 가상환경**을 만들고 의존성을 설치합니다.
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   - GPU를 쓰려면 `pip install -r requirements.txt` 전에 사용 중인 CUDA 버전에 맞는 PyTorch를 [pytorch.org](https://pytorch.org/get-started/locally/) 안내대로 먼저 설치하세요. GPU가 없으면 CPU로 동작합니다.
   - LangChain은 0.3 계열과 1.x 계열(`langchain-community`는 0.4) 모두에서 동작을 확인했으며, `requirements.txt`는 확인한 범위(다음 메이저 미만)로 제한합니다.
   - 첫 실행 때 Hugging Face에서 임베딩 모델(multilingual-e5-large, 약 2GB)을 내려받습니다.
3. `lags5.py`의 웹 수집은 requests + BeautifulSoup만 씁니다. 코드 상단에서 `playwright`, `trafilatura`, `newspaper`를 import하려고 시도하지만, `lags5.py`에서는 이 패키지들을 실제로 쓰지 않으므로 설치하지 않아도 됩니다. `legacy/rag.py`만 Playwright·trafilatura·newspaper3k를 사용합니다.

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

- Ollama 주소는 `OLLAMA_HOST` 환경변수로만 바꿀 수 있습니다(UI 설정 없음). 서버 바인딩용으로 `OLLAMA_HOST=0.0.0.0`을 설정해 둔 PC에서는 앱도 `http://0.0.0.0:11434`로 접속하며, 이 주소는 Windows에서 실패할 수 있으므로 앱 실행 시에는 `localhost` 등 실제 주소를 지정하세요.
- 웹 수집은 기본으로 robots.txt를 따릅니다(`respect_robots_txt=True`, RFC 9309 기준: robots.txt가 4xx면 전체 허용, 5xx·연결 실패면 전체 금지). UI에는 이 옵션을 끄는 칸이 없고, 끄려면 코드에서 `configure_scraping_limits(respect_robots_txt=False)`를 호출해야 합니다. `Crawl-delay`는 반영하지 않으며(요청 간격은 `delay_between_requests`), User-Agent가 브라우저 값이라 `User-agent: *` 규칙이 적용됩니다. robots.txt와 별개로 수집 대상 사이트의 이용 조건은 직접 확인하세요.
- 테스트(`tests/`)는 torch·sentence-transformers·Ollama 없이 가짜 임베딩과 가짜 LLM으로 돕니다. import, UI 구성, `OLLAMA_HOST`, 질문당 검색 1회, 벡터스토어 재시작, robots.txt를 확인하며, 실제 임베딩 모델과 답변 품질은 확인하지 않습니다. GitHub Actions가 LangChain 0.3·1.x 두 계열에서 실행합니다(`pip install -r requirements.txt pytest -c tests/constraints-langchain-1.x.txt` 후 `python -m pytest -q`, torch 없이 돌리려면 `sentence-transformers` 줄을 빼고 설치).
