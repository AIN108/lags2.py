"""torch·임베딩 모델·Ollama 없이 lags5.py를 검증하기 위한 공통 설정.

sentence-transformers(torch 포함, 수 GB)는 설치하지 않으므로, 없을 때만 빈 모듈로 대체한다.
임베딩은 langchain_core의 DeterministicFakeEmbedding, LLM은 RunnableLambda로 대신한다.
"""
import importlib.util
import runpy
import sys
import types
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if importlib.util.find_spec("sentence_transformers") is None:
    stub = types.ModuleType("sentence_transformers")
    stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = stub


def app_files():
    """실행 파일 목록. 최신판 lags5.py와 옛 사본(루트 또는 legacy/)을 모두 찾는다."""
    return sorted(p for p in [*ROOT.glob("*.py"), *ROOT.glob("legacy/*.py")])


@pytest.fixture(scope="session")
def lags5():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return runpy.run_path(str(ROOT / "lags5.py"), run_name="lags5_under_test")
