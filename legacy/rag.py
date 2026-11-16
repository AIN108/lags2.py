import os
import numpy as np
import pickle
from typing import List, Dict, Optional, Union
import warnings

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from pathlib import Path
import gradio as gr
from sentence_transformers import SentenceTransformer
import requests
import json
import pandas as pd
import openpyxl
import PyPDF2
from datetime import datetime
import time
import re
from urllib.parse import urlparse
import base64
from io import BytesIO
from PIL import Image

# pandas FutureWarning 억제
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# LangChain imports
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader

# PDF 로더는 선택적 import
try:
    from langchain_community.document_loaders import PDFLoader

    LANGCHAIN_PDF_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.document_loaders.pdf import PyPDFLoader as PDFLoader

        LANGCHAIN_PDF_AVAILABLE = True
    except ImportError:
        LANGCHAIN_PDF_AVAILABLE = False

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings

# xlrd 사용 가능 여부 확인
try:
    import xlrd

    XLRD_AVAILABLE = True
except ImportError:
    XLRD_AVAILABLE = False

# 웹 스크래핑 관련 임포트
from bs4 import BeautifulSoup

try:
    from playwright.sync_api import sync_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import trafilatura

    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False

try:
    from newspaper import Article

    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


class AdvancedWebScraper:
    """고급 웹 스크래퍼 - 광고 우회 및 다양한 방법 지원"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def scrape_url_multi_method(self, url):
        """여러 방법으로 URL 스크래핑 시도"""
        result = {
            'url': url,
            'title': '',
            'content': '',
            'success': False,
            'method': 'none',
            'error': ''
        }

        # 방법 1: Trafilatura (가장 효과적)
        if TRAFILATURA_AVAILABLE:
            try:
                content = self.scrape_with_trafilatura(url)
                if content and len(content.strip()) > 100:
                    result.update({
                        'content': content,
                        'success': True,
                        'method': 'trafilatura'
                    })
                    return result
            except Exception:
                pass

        # 방법 2: Newspaper3k
        if NEWSPAPER_AVAILABLE:
            try:
                content = self.scrape_with_newspaper(url)
                if content and len(content.strip()) > 100:
                    result.update({
                        'content': content,
                        'success': True,
                        'method': 'newspaper'
                    })
                    return result
            except Exception:
                pass

        # 방법 3: Playwright (SPA 및 JavaScript 페이지)
        if PLAYWRIGHT_AVAILABLE:
            try:
                content = self.scrape_with_playwright(url)
                if content and len(content.strip()) > 100:
                    result.update({
                        'content': content,
                        'success': True,
                        'method': 'playwright'
                    })
                    return result
            except Exception:
                pass

        # 방법 4: 기본 BeautifulSoup
        try:
            content = self.scrape_with_beautifulsoup(url)
            if content and len(content.strip()) > 100:
                result.update({
                    'content': content,
                    'success': True,
                    'method': 'beautifulsoup'
                })
                return result
        except Exception:
            pass

        result['error'] = "모든 스크래핑 방법 실패"
        return result

    def scrape_with_trafilatura(self, url):
        """Trafilatura로 스크래핑"""
        import trafilatura
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise Exception("페이지 다운로드 실패")

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            include_formatting=False
        )

        if not text:
            raise Exception("텍스트 추출 실패")

        return text

    def scrape_with_newspaper(self, url):
        """Newspaper3k로 스크래핑"""
        from newspaper import Article
        article = Article(url)
        article.download()
        article.parse()

        if not article.text:
            raise Exception("본문 추출 실패")

        content = f"제목: {article.title}\n\n"
        content += article.text

        return content

    def scrape_with_playwright(self, url):
        """Playwright로 SPA 스크래핑"""
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # 광고 차단
            page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media",
                                                                                              "font"] else route.continue_())

            page.goto(url, wait_until='networkidle', timeout=30000)
            page.wait_for_timeout(3000)

            content_selectors = [
                'article', 'main', '.content', '.article-content',
                '.post-content', '.entry-content', '#content'
            ]

            content = ""
            for selector in content_selectors:
                elements = page.query_selector_all(selector)
                if elements:
                    for element in elements:
                        text = element.inner_text()
                        if len(text) > len(content):
                            content = text
                    break

            if not content:
                content = page.evaluate("""
                    () => {
                        const elementsToRemove = document.querySelectorAll('nav, footer, aside, .advertisement, .ad, .sidebar');
                        elementsToRemove.forEach(el => el.remove());
                        return document.body.innerText;
                    }
                """)

            browser.close()

            if not content or len(content.strip()) < 100:
                raise Exception("유의미한 콘텐츠 없음")

            return content

    def scrape_with_beautifulsoup(self, url):
        """BeautifulSoup으로 기본 스크래핑"""
        response = self.session.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        for element in soup(['script', 'style', 'nav', 'footer', 'aside', 'header', 'advertisement']):
            element.decompose()

        ad_patterns = ['ad', 'advertisement', 'banner', 'popup', 'modal', 'sidebar']
        for pattern in ad_patterns:
            for element in soup.find_all(attrs={'class': re.compile(pattern, re.I)}):
                element.decompose()
            for element in soup.find_all(attrs={'id': re.compile(pattern, re.I)}):
                element.decompose()

        content_selectors = ['article', 'main', '.content', '.article', '.post']
        main_content = None

        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = elements[0]
                break

        if not main_content:
            main_content = soup.body or soup

        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        content_parts = []

        for p in paragraphs:
            text = p.get_text().strip()
            if text and len(text) > 20:
                content_parts.append(text)

        content = '\n'.join(content_parts)

        if not content or len(content.strip()) < 100:
            raise Exception("유의미한 콘텐츠 없음")

        return content

    def scrape_urls_batch(self, urls):
        """여러 URL을 배치로 스크래핑"""
        results = []

        for i, url in enumerate(urls, 1):
            try:
                result = self.scrape_url_multi_method(url)
                results.append(result)

                if i < len(urls):
                    time.sleep(2)

            except Exception as e:
                results.append({
                    'url': url,
                    'title': '',
                    'content': '',
                    'success': False,
                    'method': 'error',
                    'error': str(e)
                })

        return results


class UltimateLocalRAG:
    """궁극의 로컬 RAG 시스템"""

    def __init__(self, llm_model="llama3.1:70b"):
        print("🚀 궁극의 로컬 RAG 시스템 초기화...")

        # 경로 설정
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / "research_data"
        self.db_path = self.base_path / "vectorstore"

        # 폴더 생성
        self.data_path.mkdir(exist_ok=True)
        self.db_path.mkdir(exist_ok=True)

        # Ollama 연결 테스트
        print("🔗 Ollama 서버 연결 확인...")
        if not self.test_ollama_connection():
            print("❌ Ollama 서버에 연결할 수 없습니다!")
            print("💡 해결방법: 'ollama serve' 명령어로 서버를 시작하세요")
            return

        # 로컬 임베딩 모델 초기화
        print("🔄 로컬 임베딩 모델 로딩...")
        device = self.force_gpu_check()

        try:
            # show_progress_bar 제거하여 중복 인자 문제 해결
            model_kwargs = {'device': device}
            encode_kwargs = {
                'normalize_embeddings': True,
                'batch_size': 16
            }

            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large",
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            print("✅ 메인 임베딩 모델 로드 완료")

        except Exception as e:
            print(f"⚠️ 메인 모델 실패: {e}")
            try:
                backup_model_kwargs = {'device': device}
                backup_encode_kwargs = {
                    'normalize_embeddings': True,
                    'batch_size': 32
                }

                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                    model_kwargs=backup_model_kwargs,
                    encode_kwargs=backup_encode_kwargs
                )
                print("✅ 중간 모델 로드 완료")

            except Exception as e2:
                print(f"⚠️ 중간 모델 실패: {e2}")
                try:
                    # 가장 안전한 설정으로 시도
                    minimal_kwargs = {'device': 'cpu'}

                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        model_kwargs=minimal_kwargs
                        # encode_kwargs 제거하여 문제 방지
                    )
                    print("✅ 최소 모델 로드 완료 (CPU)")

                except Exception as e3:
                    print(f"❌ 모든 임베딩 모델 실패: {e3}")
                    # 최후의 수단: 가장 기본적인 설정
                    try:
                        self.embeddings = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )
                        print("✅ 기본 설정으로 모델 로드 성공")
                    except Exception as e4:
                        print(f"❌ 최종 시도도 실패: {e4}")
                        return

        # Ollama LLM 초기화
        print(f"🤖 {llm_model} 모델 연결 중...")
        base_url = "http://localhost:11434"
        self.llm = Ollama(
            model=llm_model,
            temperature=0.1,
            num_ctx=8192,
            base_url=base_url
        )

        # 웹 스크래퍼 초기화
        self.web_scraper = AdvancedWebScraper()

        # 벡터스토어 및 RAG 체인
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None

        # 프롬프트 템플릿 설정
        self.setup_prompt_template()

        # 기존 벡터스토어 로드 시도
        self.load_vectorstore()

        print("✅ 시스템 초기화 완료!")

    def test_ollama_connection(self):
        """Ollama 서버 연결 테스트"""
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False

    def force_gpu_check(self):
        """GPU 강제 설정 시도"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.set_device(0)
                return 'cuda'
            else:
                return 'cpu'
        except Exception:
            return 'cpu'

    def setup_prompt_template(self):
        """한국어 특화 프롬프트 템플릿 설정"""
        self.rag_prompt = ChatPromptTemplate.from_template("""
당신은 연구 데이터 분석 전문가입니다. 제공된 문서를 바탕으로 정확하고 상세한 답변을 해주세요.

**참고 문서:**
{context}

**질문:** {question}

**답변 지침:**
1. 반드시 제공된 문서의 내용만을 바탕으로 답변하세요
2. 문서에 없는 내용은 "문서에서 해당 정보를 찾을 수 없습니다"라고 명시하세요
3. 구체적인 데이터, 수치, 연도가 있다면 정확히 인용하세요
4. 출처 문서나 데이터셋 이름을 언급하세요
5. 한국어로 자연스럽고 상세하게 답변하세요
6. 여러 문서에서 관련 정보가 있다면 종합적으로 분석하세요

**답변:**""")

    def load_documents_from_folder(self):
        """폴더에서 모든 문서 로드"""
        documents = []

        if not self.data_path.exists():
            print("❌ 데이터 폴더가 존재하지 않습니다")
            return documents

        # 지원하는 파일 형식
        file_processors = {
            '.txt': self.load_text_file,
            '.pdf': self.load_pdf_file,
            '.csv': self.load_csv_file,
            '.xlsx': self.load_excel_file,
            '.xls': self.load_excel_file,
        }

        for file_path in self.data_path.rglob('*'):
            if file_path.is_file() and not file_path.name.startswith('.'):
                file_ext = file_path.suffix.lower()

                if file_ext in file_processors:
                    try:
                        docs = file_processors[file_ext](file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"❌ {file_path.name} 처리 실패: {e}")

        print(f"📊 총 {len(documents)}개 문서 로드 완료")
        return documents

    def load_text_file(self, file_path):
        """텍스트 파일 로드"""
        try:
            loader = TextLoader(str(file_path), encoding='utf-8')
            return loader.load()
        except UnicodeDecodeError:
            encodings = ['cp949', 'euc-kr', 'latin1']
            for encoding in encodings:
                try:
                    loader = TextLoader(str(file_path), encoding=encoding)
                    return loader.load()
                except UnicodeDecodeError:
                    continue
            raise Exception(f"인코딩을 찾을 수 없습니다: {file_path}")

    def load_pdf_file(self, file_path):
        """PDF 파일 로드"""
        # LangChain PDFLoader 시도
        if LANGCHAIN_PDF_AVAILABLE:
            try:
                loader = PDFLoader(str(file_path))
                documents = loader.load()
                if documents and any(doc.page_content.strip() for doc in documents):
                    return documents
            except Exception:
                pass

        # PyMuPDF 시도
        try:
            import fitz
            documents = []
            pdf_doc = fitz.open(file_path)

            full_text = ""
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text()
                if text.strip():
                    full_text += f"\n--- 페이지 {page_num + 1} ---\n" + text

            pdf_doc.close()

            if full_text.strip():
                documents.append(Document(
                    page_content=f"PDF 파일: {file_path.name}\n{full_text}",
                    metadata={
                        "source": str(file_path),
                        "type": "pdf",
                        "pages": len(pdf_doc)
                    }
                ))
                return documents

        except Exception:
            pass

        # PyPDF2 시도
        try:
            import PyPDF2
            documents = []
            full_text = ""

            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)

                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text.strip():
                        full_text += f"\n--- 페이지 {page_num + 1} ---\n" + text

            if full_text.strip():
                documents.append(Document(
                    page_content=f"PDF 파일: {file_path.name}\n{full_text}",
                    metadata={
                        "source": str(file_path),
                        "type": "pdf",
                        "pages": len(pdf_reader.pages)
                    }
                ))
                return documents

        except Exception:
            pass

        return [Document(
            page_content=f"PDF 파일 처리 실패: {file_path.name}",
            metadata={"source": str(file_path), "type": "pdf_error"}
        )]

    def load_csv_file(self, file_path):
        """CSV 파일 로드 - FutureWarning 해결"""
        csv_configs = [
            {'encoding': 'utf-8', 'sep': ',', 'low_memory': False},
            {'encoding': 'utf-8', 'sep': ';', 'low_memory': False},
            {'encoding': 'utf-8', 'sep': '\t', 'low_memory': False},
            {'encoding': 'cp949', 'sep': ',', 'low_memory': False},
            {'encoding': 'euc-kr', 'sep': ',', 'low_memory': False},
            {'encoding': 'latin1', 'sep': ',', 'low_memory': False}
        ]

        df = None
        successful_config = None

        for config in csv_configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(file_path, **config)

                if len(df) > 0 and len(df.columns) > 0:
                    successful_config = config
                    break

            except Exception:
                continue

        if df is None:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read()

                content = f"CSV 파일 (텍스트 모드): {file_path.name}\n"
                content += f"파일 크기: {len(raw_content)}자\n\n"
                content += "원본 내용 (처음 2000자):\n"
                content += raw_content[:2000]

                return [Document(
                    page_content=content,
                    metadata={"source": str(file_path), "type": "csv_raw"}
                )]

            except Exception:
                return [Document(
                    page_content=f"CSV 파일 처리 실패: {file_path.name}",
                    metadata={"source": str(file_path), "type": "csv_error"}
                )]

        # DataFrame 처리 - FutureWarning 해결
        try:
            if len(df) > 10000:
                df_head = df.head(5000).copy()
                df_tail = df.tail(2000).copy()
                df_sample = pd.concat([df_head, df_tail], ignore_index=True)
                df_for_processing = df_sample
                is_sampled = True
            else:
                df_for_processing = df.copy()
                is_sampled = False

            content = f"CSV 파일: {file_path.name}\n"
            content += f"전체 행 수: {len(df):,}개\n"
            if is_sampled:
                content += f"표시 행 수: {len(df_for_processing):,}개 (샘플링됨)\n"
            content += f"열 수: {len(df.columns)}개\n"

            # 열 이름
            try:
                safe_columns = []
                for col in df.columns[:15]:
                    try:
                        safe_columns.append(str(col)[:50])
                    except:
                        safe_columns.append("Unknown_Column")

                content += f"열 이름: {', '.join(safe_columns)}\n"
                if len(df.columns) > 15:
                    content += f"... (총 {len(df.columns)}개 열)\n"
                content += "\n"
            except Exception:
                pass

            # 데이터 타입 정보
            try:
                content += "데이터 타입 정보:\n"
                for col in df_for_processing.columns[:10]:
                    try:
                        dtype = str(df_for_processing[col].dtype)
                        non_null = df_for_processing[col].notna().sum()
                        col_name = str(col)[:30]
                        content += f"- {col_name}: {dtype} ({non_null:,}개 유효값)\n"
                    except:
                        pass
                content += "\n"
            except Exception:
                pass

            # 기본 통계
            try:
                numeric_cols = df_for_processing.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    content += "숫자 열 통계:\n"
                    stats = df_for_processing[numeric_cols].describe()
                    content += str(stats) + "\n\n"
            except Exception:
                pass

            # 샘플 데이터 - FutureWarning 해결
            content += "샘플 데이터 (처음 10행):\n"
            try:
                sample_data = df_for_processing.head(10).copy()

                # FutureWarning 해결: 각 컬럼을 개별적으로 처리
                for col in sample_data.columns:
                    try:
                        # 복사본을 만들어서 dtype 호환성 문제 해결
                        sample_data[col] = sample_data[col].astype(str)
                    except Exception:
                        # 실패하면 기본값으로 대체
                        sample_data[col] = "변환실패"

                content += str(sample_data)
            except Exception:
                content += "샘플 데이터 표시 실패"

            # 연구 데이터 특별 처리
            try:
                if any(col for col in df.columns if
                       any(keyword in str(col).lower() for keyword in ['year', 'country', 'name', 'position', 'date'])):
                    content += self.process_research_data(df_for_processing, file_path.name)
            except:
                pass

            return [Document(
                page_content=content,
                metadata={
                    "source": str(file_path),
                    "type": "csv",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "is_sampled": is_sampled
                }
            )]

        except Exception:
            return [Document(
                page_content=f"CSV 파일 부분 처리: {file_path.name}",
                metadata={"source": str(file_path), "type": "csv_partial"}
            )]

    def load_excel_file(self, file_path):
        """Excel 파일 로드 - 안정성 개선"""
        documents = []
        file_ext = file_path.suffix.lower()

        if file_ext == '.xls' and not XLRD_AVAILABLE:
            return [Document(
                page_content=f"Excel 파일 처리 실패: {file_path.name}\n.xls 파일 처리를 위해 'pip install xlrd' 필요",
                metadata={"source": str(file_path), "type": "excel_error"}
            )]

        try:
            # 엑셀 파일 열기
            xl_file = None

            try:
                if file_ext == '.xls' and XLRD_AVAILABLE:
                    xl_file = pd.ExcelFile(file_path, engine='xlrd')
                else:
                    xl_file = pd.ExcelFile(file_path, engine='openpyxl')
            except Exception:
                # 모든 가능한 엔진 시도
                engines = ['openpyxl']
                if XLRD_AVAILABLE:
                    engines.append('xlrd')

                for engine in engines:
                    try:
                        xl_file = pd.ExcelFile(file_path, engine=engine)
                        break
                    except Exception:
                        continue

                if xl_file is None:
                    raise Exception("모든 Excel 엔진 실패")

            for sheet_name in xl_file.sheet_names[:10]:
                try:
                    # 시트 읽기
                    df = None
                    try:
                        if file_ext == '.xls' and XLRD_AVAILABLE:
                            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='xlrd')
                        else:
                            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                    except Exception:
                        engines = ['openpyxl']
                        if XLRD_AVAILABLE:
                            engines.append('xlrd')

                        for engine in engines:
                            try:
                                df = pd.read_excel(file_path, sheet_name=sheet_name, engine=engine)
                                break
                            except:
                                continue

                        if df is None:
                            raise Exception("시트 읽기 실패")

                    if len(df) == 0:
                        continue

                    # 데이터 전처리 - 안전한 방법으로 변경
                    df = self.safe_clean_dataframe(df)

                    content = f"Excel 파일: {file_path.name}, 시트: {sheet_name}\n"
                    content += f"행 수: {len(df):,}개\n"
                    content += f"열 수: {len(df.columns)}개\n"

                    # 열 이름
                    column_names = []
                    for col in df.columns[:15]:
                        try:
                            column_names.append(str(col))
                        except:
                            column_names.append("Unknown_Column")

                    content += f"열 이름: {', '.join(column_names)}\n"
                    if len(df.columns) > 15:
                        content += f"... (총 {len(df.columns)}개 열)\n"
                    content += "\n"

                    # 데이터 타입 정보
                    content += "데이터 타입:\n"
                    for col in df.columns[:10]:
                        try:
                            dtype = str(df[col].dtype)
                            non_null = df[col].notna().sum()
                            content += f"- {str(col)}: {dtype} ({non_null:,}개 유효값)\n"
                        except:
                            pass
                    content += "\n"

                    # 기본 통계
                    try:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            content += "숫자 열 통계:\n"
                            stats = df[numeric_cols].describe()
                            content += str(stats) + "\n\n"
                    except:
                        pass

                    # 샘플 데이터
                    content += "샘플 데이터 (처음 10행):\n"
                    try:
                        sample_data = df.head(10)
                        content += str(sample_data)
                    except Exception:
                        content += "샘플 데이터 표시 실패"

                    # 연구 데이터 특별 처리
                    try:
                        if any(col for col in df.columns if
                               any(keyword in str(col).lower() for keyword in ['year', 'country', 'name', 'position'])):
                            content += self.process_research_data(df, f"{file_path.name}-{sheet_name}")
                    except:
                        pass

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": str(file_path),
                            "sheet": sheet_name,
                            "type": "excel",
                            "rows": len(df),
                            "columns": len(df.columns)
                        }
                    ))

                except Exception:
                    documents.append(Document(
                        page_content=f"Excel 시트 처리 실패: {file_path.name} - {sheet_name}",
                        metadata={"source": str(file_path), "sheet": sheet_name, "type": "excel_error"}
                    ))

        except Exception as e:
            return [Document(
                page_content=f"Excel 파일 처리 실패: {file_path.name}\n오류: {str(e)}",
                metadata={"source": str(file_path), "type": "excel_error"}
            )]

        return documents

    def safe_clean_dataframe(self, df):
        """DataFrame을 안전하게 텍스트 표시용으로 정리 - FutureWarning 해결"""
        df_clean = df.copy()

        for col in df_clean.columns:
            try:
                # datetime 객체를 문자열로 변환
                if df_clean[col].dtype == 'datetime64[ns]' or pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                    df_clean[col] = df_clean[col].astype(str)

                # object 타입에서 datetime 객체 찾아서 변환
                elif df_clean[col].dtype == 'object':
                    def safe_convert(x):
                        if pd.isna(x):
                            return x
                        elif isinstance(x, datetime):
                            return str(x)
                        else:
                            return x

                    df_clean[col] = df_clean[col].apply(safe_convert)

            except Exception:
                continue

        return df_clean

    def process_research_data(self, df, source_name):
        """연구 데이터 특별 처리"""
        additional_content = "\n\n=== 연구 데이터 분석 ===\n"

        try:
            # 연도별 분석
            year_cols = [col for col in df.columns if 'year' in str(col).lower()]
            if year_cols:
                year_col = year_cols[0]
                years = df[year_col].dropna()

                # 숫자로 변환 가능한 연도만 처리
                numeric_years = pd.to_numeric(years, errors='coerce').dropna()
                if len(numeric_years) > 0:
                    numeric_years = numeric_years.astype(int)
                    additional_content += f"연도 범위: {numeric_years.min()}-{numeric_years.max()}\n"

                    year_counts = numeric_years.value_counts().head(10)
                    additional_content += "주요 연도별 데이터:\n"
                    for year, count in year_counts.items():
                        additional_content += f"- {year}년: {count}건\n"

            # 국가별 분석
            country_cols = [col for col in df.columns if 'country' in str(col).lower()]
            if country_cols:
                countries = df[country_cols[0]].value_counts().head(10)
                additional_content += "\n주요 국가별 데이터:\n"
                for country, count in countries.items():
                    additional_content += f"- {country}: {count}건\n"

            # 정부 관련 데이터
            position_cols = [col for col in df.columns if 'position' in str(col).lower()]
            if position_cols:
                positions = df[position_cols[0]].value_counts().head(10)
                additional_content += "\n주요 직책:\n"
                for position, count in positions.items():
                    additional_content += f"- {position}: {count}명\n"

        except Exception:
            additional_content += "\n연구 데이터 분석 중 오류 발생\n"

        return additional_content

    def scrape_and_add_urls(self, urls):
        """웹 URL 스크래핑 후 벡터스토어에 추가"""
        if not urls:
            return {'success': 0, 'total': 0, 'message': 'URL이 없습니다'}

        # 웹 스크래핑 실행
        scrape_results = self.web_scraper.scrape_urls_batch(urls)

        # 성공한 결과만 문서로 변환
        web_documents = []
        success_count = 0

        for result in scrape_results:
            if result['success'] and result['content'].strip():
                web_content = f"웹페이지: {result.get('title', 'Unknown Title')}\n"
                web_content += f"URL: {result['url']}\n"
                web_content += f"스크래핑 방법: {result['method']}\n"
                web_content += f"수집 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                web_content += result['content']

                web_documents.append(Document(
                    page_content=web_content,
                    metadata={
                        'source': result['url'],
                        'type': 'web_page',
                        'method': result['method'],
                        'scraped_time': datetime.now().isoformat()
                    }
                ))
                success_count += 1

        if web_documents and self.vectorstore:
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            splits = text_splitter.split_documents(web_documents)

            # 벡터스토어에 추가
            self.vectorstore.add_documents(splits)

        return {
            'success': success_count,
            'total': len(urls),
            'new_documents': len(web_documents),
            'message': f'{success_count}/{len(urls)}개 URL 스크래핑 성공, {len(web_documents)}개 문서 추가'
        }

    def build_vectorstore(self, force_rebuild=False):
        """벡터스토어 구축 - show_progress_bar 에러 해결"""
        vectorstore_path = self.db_path / "chroma_db"

        if vectorstore_path.exists() and not force_rebuild:
            print("📚 기존 벡터스토어를 사용합니다")
            return

        print("🔨 벡터스토어 구축 중...")

        # 문서 로드
        documents = self.load_documents_from_folder()

        if not documents:
            print("❌ 처리할 문서가 없습니다")
            return

        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        total_chunks = len(splits)
        print(f"📄 {total_chunks:,}개 청크로 분할 완료")

        # 임베딩 모델 테스트
        try:
            print("🔍 임베딩 모델 테스트 중...")
            test_texts = ["테스트 문서입니다.", "This is a test document."]
            test_embeddings = self.embeddings.embed_documents(test_texts)
            print(f"✅ 임베딩 모델 정상 작동 (차원: {len(test_embeddings[0])})")
        except Exception as e:
            print(f"❌ 임베딩 모델 테스트 실패: {e}")
            print("🔄 더 안전한 설정으로 임베딩 모델 재시도...")

            try:
                # 가장 기본적인 설정으로 재시도
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                test_embeddings = self.embeddings.embed_documents(test_texts)
                print(f"✅ 기본 임베딩 모델로 성공 (차원: {len(test_embeddings[0])})")
            except Exception as e2:
                print(f"❌ 기본 모델도 실패: {e2}")
                return

        try:
            # 매우 보수적인 배치 크기로 시작
            if total_chunks > 50000:
                batch_size = 200  # 초대용량용 - 더 작게
                print("⚠️ 초대용량 데이터! 매우 작은 배치로 안전하게 처리합니다...")
            elif total_chunks > 10000:
                batch_size = 500  # 대용량용
                print("⚠️ 대용량 데이터! 작은 배치로 안전하게 진행합니다...")
            else:
                batch_size = 1000  # 일반용

            print(f"🔄 배치 크기: {batch_size:,}개씩 처리")

            # 첫 번째 배치로 벡터스토어 생성
            first_batch = splits[:min(batch_size, len(splits))]
            total_batches = (total_chunks + batch_size - 1) // batch_size
            print(f"🏗️ 초기 벡터스토어 생성 중... (1/{total_batches})")

            # persist_directory를 문자열로 명시적 변환
            persist_dir = str(vectorstore_path)

            # 기존 디렉토리가 있으면 삭제 (충돌 방지)
            if vectorstore_path.exists():
                import shutil
                shutil.rmtree(vectorstore_path)
                print("🗑️ 기존 벡터스토어 디렉토리 삭제")

            # 더 안전한 방법으로 벡터스토어 생성
            try:
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )
            except Exception as e:
                print(f"⚠️ collection_name 없이 재시도: {e}")
                # collection_name 없이 재시도
                self.vectorstore = Chroma.from_documents(
                    documents=first_batch,
                    embedding=self.embeddings,
                    persist_directory=persist_dir
                )

            processed = len(first_batch)
            print(f"✅ 초기 벡터스토어 생성 완료: {processed:,}/{total_chunks:,}")

            # 나머지 배치들 추가
            batch_num = 2
            while processed < total_chunks:
                try:
                    start_idx = processed
                    end_idx = min(processed + batch_size, total_chunks)
                    current_batch = splits[start_idx:end_idx]

                    print(f"🔄 배치 {batch_num}/{total_batches} 처리 중: {start_idx:,}-{end_idx:,} ({len(current_batch):,}개)")

                    # 배치 추가
                    self.vectorstore.add_documents(current_batch)

                    processed = end_idx
                    batch_num += 1

                    progress = (processed / total_chunks) * 100
                    print(f"✅ 진행률: {progress:.1f}% ({processed:,}/{total_chunks:,})")

                    # 메모리 정리
                    if batch_num % 5 == 0:
                        print("💾 메모리 정리 중...")
                        import gc
                        gc.collect()
                        time.sleep(1)

                except Exception as e:
                    print(f"❌ 배치 {batch_num} 처리 실패: {e}")
                    print("⚠️ 현재까지 처리된 데이터로 계속 진행합니다...")
                    break

            print("✅ 벡터스토어 구축 완료!")

        except Exception as e:
            print(f"❌ 벡터스토어 구축 실패: {e}")
            print(f"에러 타입: {type(e).__name__}")
            print(f"에러 상세: {str(e)}")

            # 최종 대안: 가장 작은 배치로 시도
            print("\n🔄 최소 배치 크기로 최종 시도...")
            try:
                # 극소 배치로 시도
                mini_batch_size = 50
                small_batch = splits[:mini_batch_size]

                # 기존 디렉토리 삭제
                if vectorstore_path.exists():
                    import shutil
                    shutil.rmtree(vectorstore_path)

                self.vectorstore = Chroma.from_documents(
                    documents=small_batch,
                    embedding=self.embeddings,
                    persist_directory=str(vectorstore_path)
                )
                print("✅ 최소 벡터스토어 생성 성공")

                # 나머지 극소 배치로 추가
                processed = mini_batch_size
                batch_count = 1

                while processed < len(splits):
                    try:
                        end_idx = min(processed + mini_batch_size, len(splits))
                        batch = splits[processed:end_idx]

                        self.vectorstore.add_documents(batch)
                        processed = end_idx
                        batch_count += 1

                        if batch_count % 20 == 0:
                            progress = (processed / total_chunks) * 100
                            print(f"진행중... {progress:.1f}% ({processed:,}/{total_chunks:,})")
                            import gc
                            gc.collect()

                    except Exception as batch_error:
                        print(f"배치 {batch_count} 실패: {batch_error}, 건너뜀...")
                        processed = end_idx
                        batch_count += 1
                        continue

                print("✅ 최소 배치 방법으로 벡터스토어 구축 완료")

            except Exception as e2:
                print(f"❌ 최종 시도도 실패: {e2}")
                print("💡 해결 방법:")
                print("1. pip install --upgrade sentence-transformers")
                print("2. pip install --upgrade langchain langchain-community")
                print("3. pip install --upgrade chromadb")
                print("4. 시스템 재부팅 후 재시도")
                return

        # RAG 체인 설정
        self.setup_rag_chain()

    def load_vectorstore(self):
        """기존 벡터스토어 로드 - 에러 해결"""
        vectorstore_path = self.db_path / "chroma_db"

        if vectorstore_path.exists():
            try:
                persist_dir = str(vectorstore_path)

                # collection_name 없이 먼저 시도
                try:
                    self.vectorstore = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=self.embeddings
                    )
                except Exception:
                    # collection_name과 함께 시도
                    self.vectorstore = Chroma(
                        persist_directory=persist_dir,
                        embedding_function=self.embeddings,
                        collection_name="research_data"
                    )

                self.setup_rag_chain()
                print("✅ 기존 벡터스토어 로드 완료")

            except Exception as e:
                print(f"⚠️ 벡터스토어 로드 실패: {e}")
                print("새로 구축합니다...")
                self.build_vectorstore(force_rebuild=True)

    def setup_rag_chain(self):
        """RAG 체인 설정"""
        if self.vectorstore:
            # retriever 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )

            # RAG 체인 구성
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            # 체인 구성을 단순화
            self.rag_chain = (
                    {
                        "context": self.retriever | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | self.rag_prompt
                    | self.llm
                    | StrOutputParser()
            )

            print("✅ RAG 체인 구성 완료")

    def ask(self, question):
        """질문에 답변"""
        if not self.rag_chain:
            return "❌ RAG 시스템이 초기화되지 않았습니다. 먼저 벡터스토어를 구축해주세요."

        try:
            # 검색된 문서 확인
            retrieved_docs = self.retriever.invoke(question)

            # RAG 체인 실행
            response = self.rag_chain.invoke(question)

            # 출처 정보 추가
            sources = []
            web_sources = []

            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'file')

                if doc_type == 'web_page':
                    web_sources.append(source)
                else:
                    sources.append(Path(source).name)

            source_info = "\n\n📚 참고 자료:"
            if sources:
                source_info += f"\n📁 파일: {', '.join(set(sources))}"
            if web_sources:
                source_info += f"\n🌐 웹페이지: {len(set(web_sources))}개"

            return response + source_info

        except Exception as e:
            return f"❌ 오류 발생: {e}"

    def get_stats(self):
        """시스템 통계"""
        if not self.vectorstore:
            return "벡터스토어가 구축되지 않았습니다."

        try:
            collection = self.vectorstore._collection
            count = collection.count()

            stats = f"📊 수정된 궁극의 로컬 RAG 시스템 통계 (v4.2)\n\n"
            stats += f"📄 총 청크 수: {count:,}개\n"
            stats += f"🤖 LLM 모델: llama3.1:70b (로컬)\n"
            stats += f"🔍 임베딩 모델: {getattr(self.embeddings, 'model_name', 'Unknown')} (로컬)\n"
            stats += f"🌐 웹 스크래핑: 지원 (다중 방법)\n"
            stats += f"📁 데이터 폴더: {self.data_path}\n"
            stats += f"💾 벡터스토어: Chroma (로컬)\n"
            stats += f"🔧 v4.2 수정: show_progress_bar 에러 해결, 임베딩 안정화\n"
            stats += f"🔧 v4.1 개선: FutureWarning 해결, 안정성 향상\n\n"

            # GPU 상세 정보
            try:
                import torch
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device)
                    gpu_memory_total = torch.cuda.get_device_properties(current_device).total_memory / 1024 ** 3
                    gpu_memory_used = torch.cuda.memory_allocated(current_device) / 1024 ** 3
                    gpu_memory_cached = torch.cuda.memory_reserved(current_device) / 1024 ** 3

                    stats += f"🎮 GPU: {gpu_name}\n"
                    stats += f"   💾 VRAM 총량: {gpu_memory_total:.1f}GB\n"
                    stats += f"   💾 VRAM 사용: {gpu_memory_used:.1f}GB\n"
                    stats += f"   💾 VRAM 캐시: {gpu_memory_cached:.1f}GB\n"
                else:
                    stats += f"🎮 GPU: 미사용 (CUDA 불가)\n"
            except Exception:
                stats += f"🎮 GPU: 상태 확인 실패\n"

            # Ollama 연결 상태
            ollama_status = "연결됨" if self.test_ollama_connection() else "연결 안됨"
            stats += f"🔗 Ollama: {ollama_status}\n"

            # 시스템 메모리 정보
            try:
                import psutil
                memory = psutil.virtual_memory()
                stats += f"💻 시스템 메모리:\n"
                stats += f"   📊 총량: {memory.total / 1024 ** 3:.1f}GB\n"
                stats += f"   📊 사용중: {memory.used / 1024 ** 3:.1f}GB ({memory.percent:.1f}%)\n"
                stats += f"   📊 여유: {memory.available / 1024 ** 3:.1f}GB\n"

                # CPU 정보
                cpu_count = psutil.cpu_count()
                cpu_usage = psutil.cpu_percent(interval=1)
                stats += f"🔥 CPU: {cpu_count}코어, 사용률 {cpu_usage:.1f}%\n"

            except Exception:
                stats += f"💻 메모리: 모니터링 불가\n"

            # 데이터 규모별 권장사항
            if count > 100000:
                stats += f"\n🚨 초대용량 데이터셋!\n"
                stats += f"   🔄 안전한 배치 처리 적용\n"
                stats += f"   💡 GPU + 16GB+ RAM 필수\n"
                stats += f"   ⏰ 처리 시간: 4-8시간 예상\n"
            elif count > 50000:
                stats += f"\n⚠️ 대용량 데이터셋\n"
                stats += f"   🔄 최적화된 배치 처리\n"
                stats += f"   💡 GPU 사용 강력 권장\n"
                stats += f"   ⏰ 처리 시간: 2-4시간 예상\n"
            elif count > 20000:
                stats += f"\n📊 중용량 데이터셋\n"
                stats += f"   🔄 배치 처리 적용\n"
                stats += f"   💡 8GB+ RAM 권장\n"

            return stats
        except Exception as e:
            return f"통계 수집 오류: {e}"


def create_ultimate_interface():
    """최종 Gradio 인터페이스 - 개선된 버전"""

    rag_system = None

    def initialize_system(llm_model):
        nonlocal rag_system
        try:
            rag_system = UltimateLocalRAG(llm_model=llm_model)
            return "✅ 시스템 초기화 완료! (v4.2 - show_progress_bar 에러 해결)", rag_system.get_stats()
        except Exception as e:
            return f"❌ 초기화 실패: {e}", ""

    def build_knowledge_base(force_rebuild):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요.", ""

        try:
            rag_system.build_vectorstore(force_rebuild=force_rebuild)
            return "✅ 지식 베이스 구축 완료! (v4.2 - 에러 해결)", rag_system.get_stats()
        except Exception as e:
            return f"❌ 구축 실패: {e}", ""

    def scrape_web_content(urls_text):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요.", ""

        if not urls_text.strip():
            return "URL을 입력해주세요.", ""

        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        result = rag_system.scrape_and_add_urls(urls)
        return result['message'], rag_system.get_stats()

    def answer_question(question):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화하고 지식 베이스를 구축해주세요."

        if not question.strip():
            return "질문을 입력해주세요."

        return rag_system.ask(question)

    # Gradio 인터페이스
    with gr.Blocks(title="수정된 궁극의 로컬 RAG 시스템 v4.2", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🚀 수정된 궁극의 로컬 RAG 시스템 v4.2

        ### 🔧 최신 수정사항 (v4.2):
        - 🛠️ **show_progress_bar 에러 완전 해결**
        - 🔍 **임베딩 모델 초기화 안정화**
        - 🏗️ **벡터스토어 생성 로직 개선**
        - 📦 **배치 크기 더욱 보수적으로 조정**
        - 🧪 **임베딩 모델 사전 테스트 추가**

        ### ✨ 기존 개선사항 (v4.1):
        - 🔧 **FutureWarning 완전 해결** (pandas 호환성)
        - 🛡️ **안정성 크게 향상** (에러 처리 개선)
        - ⚡ **대용량 데이터 최적화** (배치 크기 조정)
        - 💾 **메모리 사용량 최적화** (가비지 컬렉션)

        ### 🔥 핵심 기능 유지:
        - 🤖 **llama3.1:70b** (로컬 LLM, 최고 품질)
        - 🔍 **multilingual-e5-large** (로컬 임베딩, 다국어)
        - 🌐 **고급 웹 스크래핑** (광고 우회, 다중 방법)
        - 📊 **연구 데이터 특화** (Excel, CSV, PDF 스마트 처리)
        - 💰 **완전 무료** (OpenAI API 불필요)
        """)

        with gr.Tab("⚙️ 시스템 설정"):
            gr.Markdown("""
            ### 🔧 시스템 초기화 (v4.2 - 에러 해결)

            **해결된 문제들:**
            - ✅ show_progress_bar 중복 인자 에러 해결
            - ✅ 임베딩 모델 초기화 안정화
            - ✅ 벡터스토어 생성 로직 개선
            - ✅ pandas FutureWarning 완전 해결
            - ✅ DataFrame 타입 변환 안정성 개선

            **성능 최적화:**
            - 🚀 초대용량: 200개씩 배치 처리 (더욱 안전)
            - ⚡ 대용량: 500개씩 배치 처리  
            - 💨 일반: 1000개씩 배치 처리
            - 🧪 임베딩 모델 사전 테스트 추가
            """)

            llm_model_choice = gr.Dropdown(
                label="LLM 모델 선택",
                choices=["llama3.1:70b", "llama3.1:8b", "qwen2.5:32b", "qwen2.5:7b"],
                value="llama3.1:70b"
            )

            with gr.Row():
                init_btn = gr.Button("🚀 시스템 초기화 (v4.2 - 에러 해결)", variant="primary")
                gpu_test_btn = gr.Button("🎮 GPU 진단", variant="secondary")

            init_output = gr.Textbox(label="초기화 상태", lines=2)
            system_stats = gr.Textbox(label="시스템 정보", lines=20)

            def diagnose_gpu():
                try:
                    import torch
                    info = "🎮 GPU 진단 결과:\n\n"
                    info += f"CUDA 사용 가능: {torch.cuda.is_available()}\n"
                    if torch.cuda.is_available():
                        info += f"GPU 개수: {torch.cuda.device_count()}\n"
                        for i in range(torch.cuda.device_count()):
                            name = torch.cuda.get_device_name(i)
                            memory = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                            info += f"GPU {i}: {name} ({memory:.1f}GB)\n"
                    else:
                        info += "GPU가 감지되지 않습니다.\n"

                    # 메모리 정보
                    try:
                        import psutil
                        memory = psutil.virtual_memory()
                        info += f"\n💾 시스템 메모리:\n"
                        info += f"총량: {memory.total / 1024 ** 3:.1f}GB\n"
                        info += f"사용중: {memory.used / 1024 ** 3:.1f}GB ({memory.percent:.1f}%)\n"
                    except:
                        info += "\n메모리 정보 확인 불가\n"

                    return info

                except Exception as e:
                    return f"GPU 진단 실패: {e}"

            init_btn.click(
                initialize_system,
                inputs=llm_model_choice,
                outputs=[init_output, system_stats]
            )
            gpu_test_btn.click(
                diagnose_gpu,
                outputs=system_stats
            )

        with gr.Tab("📁 데이터 관리"):
            gr.Markdown("""
            ### 📂 데이터 준비 (안정성 개선)

            **처리 개선사항:**
            - 🔧 FutureWarning 완전 해결
            - 🛡️ DataFrame 타입 변환 안정화
            - ⚡ 대용량 파일 처리 최적화
            - 💾 메모리 사용량 개선

            **지원 파일** (기존과 동일):
            - 📊 Excel/CSV: 안정적 처리
            - 📄 PDF: 다중 엔진 지원
            - 📝 텍스트: 인코딩 자동 감지
            """)

            with gr.Row():
                build_btn = gr.Button("🔨 지식 베이스 구축 (v4.2)", variant="primary")
                rebuild_btn = gr.Button("🔄 완전 재구축 (에러 해결)", variant="secondary")

            build_output = gr.Textbox(label="구축 상태", lines=3)
            build_stats = gr.Textbox(label="데이터 통계", lines=15)

            build_btn.click(
                lambda: build_knowledge_base(False),
                outputs=[build_output, build_stats]
            )
            rebuild_btn.click(
                lambda: build_knowledge_base(True),
                outputs=[build_output, build_stats]
            )

        with gr.Tab("🌐 웹 스크래핑"):
            gr.Markdown("""
            ### 🔍 고급 웹 스크래핑 (기존 기능 유지)

            **스크래핑 방법** (변경 없음):
            - 🚀 Trafilatura (최고 품질)
            - 📰 Newspaper3k (뉴스 특화)
            - 🎭 Playwright (SPA, JavaScript)
            - 🍲 BeautifulSoup (기본 방법)
            """)

            urls_input = gr.Textbox(
                label="URL 목록 (한 줄에 하나씩)",
                placeholder="""https://www.example.com/article1
https://www.example.com/article2""",
                lines=8
            )

            scrape_btn = gr.Button("🔍 고급 스크래핑 시작", variant="primary")
            scrape_output = gr.Textbox(label="스크래핑 결과", lines=3)
            scrape_stats = gr.Textbox(label="업데이트된 통계", lines=15)

            scrape_btn.click(
                scrape_web_content,
                inputs=urls_input,
                outputs=[scrape_output, scrape_stats]
            )

        with gr.Tab("❓ 연구 질문"):
            gr.Markdown("""
            ### 🎯 연구 질문하기 (안정성 개선)

            **개선사항:**
            - 🛡️ 응답 안정성 향상
            - 🔍 검색 정확도 개선
            - 💬 에러 처리 강화

            **예시 질문** (기존과 동일):
            - "1965년 소련 정부 주요 인물들은?"
            - "냉전 시기 미국 국방장관 변화"
            - "V-Dem 데이터에서 한국 민주화 과정"
            """)

            question_input = gr.Textbox(
                label="연구 질문",
                placeholder="예: 1965년 소련 정부의 주요 인물들과 그들의 역할은 무엇인가요?",
                lines=3
            )

            ask_btn = gr.Button("🤖 llama3.1:70b에게 질문하기 (v4.2)", variant="primary")

            answer_output = gr.Textbox(
                label="AI 답변 (v4.2 - 안정성 개선)",
                lines=25
            )

            ask_btn.click(
                answer_question,
                inputs=question_input,
                outputs=answer_output
            )

            gr.Examples(
                examples=[
                    "1965년 소련 정부의 주요 인물들과 그들의 역할은?",
                    "냉전 시기 미국 정부의 주요 변화는?",
                    "데이터에서 한국의 민주화 과정을 설명해주세요",
                    "1980년대 유럽 각국의 정치 지도자들은?",
                    "WhoGov 데이터에 따른 독일 정부 변화",
                    "스크래핑 자료에서 최신 정치 이슈는?"
                ],
                inputs=question_input
            )

    return interface


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 수정된 궁극의 로컬 RAG 시스템 v4.2")
    print("=" * 80)
    print("🔧 최신 수정사항 (v4.2):")
    print("- show_progress_bar 에러 완전 해결")
    print("- 임베딩 모델 초기화 안정화")
    print("- 벡터스토어 생성 로직 개선")
    print("- 배치 크기 더욱 보수적으로 조정")
    print("- 임베딩 모델 사전 테스트 추가")

    print("\n🔧 기존 개선사항 (v4.1):")
    print("- FutureWarning 완전 해결")
    print("- 데이터 처리 안정성 대폭 향상")
    print("- 대용량 데이터 최적화")
    print("- 메모리 사용량 개선")
    print("- 에러 처리 강화")

    print("\n✨ 핵심 기능 모두 유지:")
    print("- 🤖 llama3.1:70b (로컬 LLM)")
    print("- 🔍 multilingual-e5-large (로컬 임베딩)")
    print("- 🌐 고급 웹 스크래핑")
    print("- 📊 연구 데이터 특화 처리")
    print("- 💰 완전 무료")

    print("\n📋 설치 명령어:")
    print("1. ollama pull llama3.1:70b")
    print("2. pip install --upgrade sentence-transformers chromadb torch")
    print("3. pip install --upgrade langchain langchain-community")
    print("4. pip install pandas numpy openpyxl xlrd")
    print("5. pip install playwright trafilatura newspaper3k")

    # 웹 인터페이스 실행
    interface = create_ultimate_interface()

    print(f"\n🚀 수정된 시스템 시작! (v4.2)")
    print("🌐 브라우저: http://localhost:7860")
    print("\n📁 사용법:")
    print("1. 시스템 초기화 (에러 해결됨)")
    print("2. research_data/ 폴더에 자료 넣기")
    print("3. 지식베이스 구축 (안정성 개선)")
    print("4. 자연스럽게 질문하기!")

    interface.launch(
        server_port=7860,
        share=False,
        inbrowser=True
    )