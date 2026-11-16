"""메인 임베딩 모델 로드 실패 시 백업 모델 대체 규칙 (가짜 임베딩, torch·Ollama 없이).

백업 모델(all-MiniLM-L6-v2, 384차원)은 메인 모델(multilingual-e5-large, 1024차원)로 만든 벡터 DB를
검색할 수 없으므로, 기존 DB가 있으면 대체하지 않고 오류를 내야 한다.
"""
import contextlib
import http.server
import io
import re
import tempfile
import threading
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from conftest import ROOT

MAIN = "intfloat/multilingual-e5-large"
BACKUP = "sentence-transformers/all-MiniLM-L6-v2"


@pytest.fixture
def init_rag(lags5, monkeypatch, tmp_path):
    """메인 모델 로드가 실패하는 환경에서 __init__을 실행한다. 반환: (rag 또는 예외, 출력, 로드 시도한 모델 목록)"""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"127.0.0.1:{server.server_port}")
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "systemp"))
    Path(tempfile.tempdir).mkdir()

    cls = lags5["UltimateMilitaryRAGv6"]
    init_globals = cls.__init__.__globals__
    loaded = []

    def fake_hf(model_name, **kwargs):
        loaded.append(model_name)
        if model_name == MAIN:
            raise OSError("모델을 내려받을 수 없습니다 (테스트)")
        return DeterministicFakeEmbedding(size=384)

    monkeypatch.setitem(init_globals, "HuggingFaceEmbeddings", fake_hf)
    monkeypatch.setitem(init_globals, "Ollama", lambda **kw: object())
    monkeypatch.setitem(init_globals, "__file__", str(tmp_path / "lags5.py"))

    def run():
        out = io.StringIO()
        rag = cls.__new__(cls)
        monkeypatch.setattr(rag, "load_vectorstore", lambda: None, raising=False)
        try:
            with contextlib.redirect_stdout(out):
                cls.__init__(rag, llm_model="qwen2.5:7b")
            result = rag
        except Exception as exc:  # noqa: BLE001 - 결과로 돌려 검사한다
            result = exc
        return result, out.getvalue(), loaded

    yield run
    server.shutdown()


@pytest.mark.parametrize("where", ["default", "alternative"])
def test_existing_db_blocks_fallback_with_clear_error(init_rag, tmp_path, where):
    db = (tmp_path / "vectorstore" if where == "default" else Path(tempfile.tempdir)) / "chroma_db_v6_military"
    db.mkdir(parents=True)
    result, _, loaded = init_rag()
    assert isinstance(result, RuntimeError), result
    message = str(result)
    assert str(db) in message
    assert "1024" in message and "384" in message
    assert "해결" in message
    assert BACKUP not in loaded  # 백업 모델을 아예 불러오지 않는다


def test_without_db_fallback_is_allowed_with_warning(init_rag):
    result, output, loaded = init_rag()
    assert not isinstance(result, Exception), result
    assert loaded == [MAIN, BACKUP]
    assert result.embeddings.size == 384
    assert re.search(r"백업 임베딩 모델\(all-MiniLM-L6-v2, 384차원\)로 대체했습니다", output)


def test_existing_db_with_main_model_is_unaffected(init_rag, tmp_path, monkeypatch, lags5):
    (tmp_path / "vectorstore" / "chroma_db_v6_military").mkdir(parents=True)
    init_globals = lags5["UltimateMilitaryRAGv6"].__init__.__globals__
    monkeypatch.setitem(init_globals, "HuggingFaceEmbeddings", lambda **kw: DeterministicFakeEmbedding(size=1024))
    result, _, _ = init_rag()
    assert not isinstance(result, Exception), result
    assert result.embeddings.size == 1024


def test_install_hint_lists_langchain_huggingface_like_requirements():
    source = (ROOT / "lags5.py").read_text(encoding="utf-8")
    requirement = next(line for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
                       if line.startswith("langchain-huggingface"))
    hints = [line for line in source.splitlines() if "pip install" in line and "print(" in line]
    assert any(f'"{requirement}"' in line for line in hints), hints
