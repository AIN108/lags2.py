"""lags5.py와 옛 사본 테스트. LangChain 0.3 계열과 1.x 계열 양쪽에서 같은 결과여야 한다."""
import contextlib
import hashlib
import http.server
import io
import re
import tempfile
import threading
import warnings
from pathlib import Path

import pytest
from conftest import app_files
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda


# --- import --------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", app_files(), ids=lambda p: p.name)
def test_import_header(path):
    """각 파일의 첫 class 앞(import 블록 전체)이 현재 LangChain 버전에서 실행되는지."""
    source = path.read_text(encoding="utf-8")
    header = source[: re.search(r"^class ", source, re.M).start()]
    namespace = {"__name__": "header_check", "__file__": str(path)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exec(compile(header, str(path), "exec"), namespace)
    assert namespace["Document"] is Document


def test_ui_builds_without_launch(lags5):
    assert type(lags5["create_v6_military_interface"]()).__name__ == "Blocks"


# --- Ollama 주소 ---------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (None, "http://localhost:11434"),
    ("0.0.0.0", "http://0.0.0.0:11434"),
    ("gpu-box", "http://gpu-box:11434"),
    ("127.0.0.1:8080", "http://127.0.0.1:8080"),
    ("http://host:1234/", "http://host:1234"),
    ("https://ollama.example.com", "https://ollama.example.com"),
])
def test_ollama_base_url(lags5, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("OLLAMA_HOST", raising=False)
    else:
        monkeypatch.setenv("OLLAMA_HOST", value)
    assert lags5["ollama_base_url"]() == expected


def test_init_uses_ollama_host(lags5, monkeypatch, tmp_path):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/api/version" else 404)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    captured = {}
    cls = lags5["UltimateMilitaryRAGv6"]
    init_globals = cls.__init__.__globals__
    monkeypatch.setitem(init_globals, "Ollama", lambda **kw: captured.update(kw))
    monkeypatch.setitem(init_globals, "HuggingFaceEmbeddings", lambda **kw: object())
    # __init__은 lags5.py 옆에 research_data/·vectorstore/를 만들므로 임시 폴더를 가리키게 한다
    monkeypatch.setitem(init_globals, "__file__", str(tmp_path / "lags5.py"))
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{server.server_port}")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rag = cls.__new__(cls)
            monkeypatch.setattr(rag, "load_vectorstore", lambda: None, raising=False)
            cls.__init__(rag, llm_model="qwen2.5:7b")
    finally:
        server.shutdown()
    assert captured["base_url"] == f"http://127.0.0.1:{server.server_port}"


# --- ask(): 검색 1회 ------------------------------------------------------------------------------

def _rag_without_init(lags5, vectorstore, llm):
    cls = lags5["UltimateMilitaryRAGv6"]
    rag = cls.__new__(cls)  # 모델 로드·Ollama 연결 생략
    rag.vectorstore = vectorstore
    rag.llm = llm
    rag.current_template = lags5["MilitaryAnalysisTemplates"]().get_military_analysis_template()
    rag.economic_processor = lags5["ImprovedEconomicDataProcessor"]()
    return rag


def test_ask_retrieves_once_and_passes_context(lags5):
    calls = {"retriever": 0}
    prompts = []
    docs = [Document(page_content="T-72B3 125mm 2A46M-5", metadata={"source": "data/tanks.txt"})]

    def retrieve(_query):
        calls["retriever"] += 1
        return docs

    class Store:
        def as_retriever(self, **kwargs):
            return RunnableLambda(retrieve)

    def llm(prompt):
        prompts.append(prompt.to_string())
        return "ANSWER-" + hashlib.sha256(prompts[-1].encode()).hexdigest()[:12]

    rag = _rag_without_init(lags5, Store(), RunnableLambda(llm))
    with contextlib.redirect_stdout(io.StringIO()):
        rag.setup_rag_chain()
    answer = rag.ask("러시아 T-72 전차 사양은?")
    assert calls["retriever"] == 1
    assert len(prompts) == 1 and "2A46M-5" in prompts[0]
    assert answer.startswith("ANSWER-") and "tanks.txt" in answer


# --- 대안 경로 벡터스토어 ------------------------------------------------------------------------

def _app(lags5, work):
    cls = lags5["UltimateMilitaryRAGv6"]
    rag = _rag_without_init(lags5, None, RunnableLambda(lambda p: "가짜 답변"))
    rag.retriever = rag.rag_chain = None
    rag.base_path, rag.data_path, rag.db_path = work, work / "research_data", work / "vectorstore"
    rag.db_path.mkdir(exist_ok=True)
    rag.embeddings = DeterministicFakeEmbedding(size=16)
    with contextlib.redirect_stdout(io.StringIO()):
        cls.load_vectorstore(rag)
    return rag


@pytest.mark.parametrize("alternative", [True, False], ids=["alternative-path", "default-path"])
def test_vectorstore_survives_restart(lags5, tmp_path, monkeypatch, alternative):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "systemp"))
    Path(tempfile.tempdir).mkdir()
    (tmp_path / "research_data").mkdir()
    (tmp_path / "research_data" / "t72.txt").write_text("T-72B3 main gun is the 125mm 2A46M-5.", "utf-8")

    first = _app(lags5, tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        assert first.build_vectorstore(use_alternative_path=alternative)
    assert first.rag_chain is not None

    restarted = _app(lags5, tmp_path)  # 앱 재시작
    assert restarted.rag_chain is not None
    with contextlib.redirect_stdout(io.StringIO()):
        assert restarted.build_vectorstore(use_alternative_path=alternative)  # '구축' 버튼 다시 누름
    assert restarted.ask("T-72B3 주포는?").startswith("가짜 답변")


# --- robots.txt ----------------------------------------------------------------------------------

ARTICLE = ("<html><head><title>{t}</title></head><body><article><h1>{t}</h1>" + "".join(
    f"<p>Paragraph {i} about armored vehicles, logistics, doctrine number {i * 7} and history volume {i}.</p>"
    for i in range(12)) + "</article></body></html>")


@pytest.fixture
def site():
    state = {"status": 200, "robots": "User-agent: *\nDisallow: /private/\n", "hits": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"].append(self.path)
            if self.path == "/robots.txt":
                self.send_response(state["status"])
                self.end_headers()
                self.wfile.write(state["robots"].encode())
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(ARTICLE.format(t=self.path.strip("/").replace("/", " ")).encode())

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", state
    server.shutdown()


@pytest.mark.parametrize("status,respect,expected", [
    (200, True, ["/public/tank-history"]),                          # Disallow 적용
    (200, False, ["/private/secret-doc", "/public/tank-history"]),  # 옵션을 끄면 기존 동작
    (404, True, ["/private/secret-doc", "/public/tank-history"]),   # robots.txt 없음 → 전체 허용
    (503, True, []),                                                # 서버 오류 → 전체 금지 (RFC 9309)
])
def test_scraper_honors_robots_txt(lags5, site, status, respect, expected):
    base, state = site
    state["status"] = status
    scraper = lags5["PersonalDataWebScraper"]()
    scraper.configure_scraping_limits(max_pages_total=10, max_depth=0, delay_between_requests=0,
                                      respect_robots_txt=respect)
    with contextlib.redirect_stdout(io.StringIO()):
        results = scraper.scrape_any_content_type([f"{base}/public/tank-history", f"{base}/private/secret-doc"])
    assert sorted(r["url"].replace(base, "") for r in results) == expected
    pages = sorted(p for p in state["hits"] if p != "/robots.txt")
    assert pages == expected  # 금지된 페이지는 요청조차 하지 않는다
