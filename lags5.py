import os
import numpy as np
import pickle
from typing import List, Dict, Optional, Union
import warnings
import hashlib
from urllib.parse import urljoin, urlparse
from collections import deque

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
import base64
from io import BytesIO
from PIL import Image

# pandas FutureWarning 억제
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# LangChain imports - 최신 버전 대응
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.documents import Document
# 임베딩·벡터 DB는 지원이 종료(sunset)된 langchain-community 대신 전용 패키지를 쓴다.
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 최신 LangChain 패키지 import
try:
    from langchain_ollama import OllamaLLM as Ollama
    OLLAMA_NEW = True
except ImportError:
    from langchain_community.llms import Ollama
    OLLAMA_NEW = False

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


def ollama_base_url():
    """OLLAMA_HOST 환경변수(Ollama CLI와 같은 변수)를 따른다. 없으면 기존 기본값 http://localhost:11434."""
    host = os.environ.get("OLLAMA_HOST", "").strip().rstrip("/")
    if not host:
        return "http://localhost:11434"
    if host.startswith(("http://", "https://")):
        return host
    # 스킴 없는 값(예: 0.0.0.0, gpu-box)은 Ollama CLI처럼 포트가 없으면 11434를 붙인다.
    parts = urlparse("//" + host)
    netloc = parts.netloc if parts.port is not None else f"{parts.netloc}:11434"
    return f"http://{netloc}{parts.path}"


class DuplicateDetector:
    """중복 콘텐츠 감지 및 제거"""

    def __init__(self, similarity_threshold=0.85):
        self.content_hashes = set()
        self.content_fingerprints = {}
        self.similarity_threshold = similarity_threshold

    def get_content_hash(self, content):
        """콘텐츠 해시 생성"""
        # 텍스트 정규화
        normalized = re.sub(r'\s+', ' ', content.lower().strip())
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return hashlib.md5(normalized.encode()).hexdigest()

    def get_content_fingerprint(self, content):
        """콘텐츠 지문 생성 (유사도 비교용)"""
        # 단어 기반 지문
        words = re.findall(r'\w+', content.lower())
        if len(words) < 10:
            return None

        # 상위 빈도 단어들로 지문 생성
        from collections import Counter
        word_freq = Counter(words)
        top_words = [word for word, _ in word_freq.most_common(50)]
        return ' '.join(sorted(top_words))

    def is_duplicate(self, content, url=None):
        """중복 여부 확인"""
        if not content or len(content.strip()) < 100:
            return True

        # 1. 정확한 해시 매칭
        content_hash = self.get_content_hash(content)
        if content_hash in self.content_hashes:
            return True

        # 2. 유사도 기반 매칭
        fingerprint = self.get_content_fingerprint(content)
        if fingerprint:
            for existing_fp in self.content_fingerprints.values():
                similarity = self.calculate_similarity(fingerprint, existing_fp)
                if similarity > self.similarity_threshold:
                    return True

        # 새 콘텐츠로 등록
        self.content_hashes.add(content_hash)
        if fingerprint and url:
            self.content_fingerprints[url] = fingerprint

        return False

    def calculate_similarity(self, text1, text2):
        """텍스트 유사도 계산"""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        if len(words1) == 0 or len(words2) == 0:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)


class PersonalDataWebScraper:
    """개인 자료 수집용 범용 웹 스크래퍼"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.duplicate_detector = DuplicateDetector()
        self.visited_urls = set()
        self.domain_stats = {}
        self.robots_cache = {}  # "scheme://host" -> RobotFileParser

        # 기본 설정 초기화
        self.config = {
            'max_pages_total': 1000,
            'max_pages_per_domain': 100,
            'max_depth': 3,
            'delay_between_requests': 1,
            'respect_robots_txt': True,
            'custom_filters': {}
        }

    def configure_scraping_limits(self, max_pages_total=1000, max_pages_per_domain=100,
                                  max_depth=3, delay_between_requests=1,
                                  respect_robots_txt=True, custom_filters=None):
        """스크래핑 제한 설정"""
        self.config = {
            'max_pages_total': max_pages_total,
            'max_pages_per_domain': max_pages_per_domain,
            'max_depth': max_depth,
            'delay_between_requests': delay_between_requests,
            'respect_robots_txt': respect_robots_txt,
            'custom_filters': custom_filters or {}
        }

    def is_allowed_by_robots(self, url):
        """robots.txt 규칙 확인 (RFC 9309). 사이트별로 한 번만 받아 캐시한다.

        4xx(파일 없음 등)는 전체 허용, 5xx·연결 실패는 전체 금지로 본다.
        """
        from urllib.robotparser import RobotFileParser

        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return True
        base = f"{parsed.scheme}://{parsed.netloc}"
        parser = self.robots_cache.get(base)
        if parser is None:
            parser = RobotFileParser()
            try:
                response = self.session.get(base + "/robots.txt", timeout=10)
                if response.status_code >= 500:
                    parser.disallow_all = True
                elif response.status_code >= 400:
                    parser.allow_all = True
                else:
                    parser.parse(response.text.splitlines())
            except requests.RequestException:
                parser.disallow_all = True
            self.robots_cache[base] = parser
        return parser.can_fetch(self.headers['User-Agent'], url)

    def scrape_any_content_type(self, start_urls, content_types='all'):
        """모든 종류의 웹페이지 콘텐츠 추출"""

        results = []
        url_queue = deque()

        # 시작 URL들 큐에 추가
        for url in start_urls:
            url_queue.append((url, 0))  # (url, depth)

        total_scraped = 0

        print(f"🔍 스크래핑 시작: {len(start_urls)}개 시작 URL")

        while url_queue and total_scraped < self.config.get('max_pages_total', 1000):
            current_url, depth = url_queue.popleft()

            # 깊이 제한 확인
            if depth > self.config.get('max_depth', 3):
                continue

            # 이미 방문한 URL 건너뛰기
            if current_url in self.visited_urls:
                continue

            # robots.txt 확인 (설정값 respect_robots_txt, 기본 True)
            if self.config.get('respect_robots_txt', True) and not self.is_allowed_by_robots(current_url):
                print(f"🤖 robots.txt에 의해 제외: {current_url}")
                self.visited_urls.add(current_url)
                continue

            # 도메인별 제한 확인
            domain = urlparse(current_url).netloc
            domain_count = self.domain_stats.get(domain, 0)

            if domain_count >= self.config.get('max_pages_per_domain', 100):
                print(f"⚠️ 도메인 제한 도달: {domain} ({domain_count}페이지)")
                continue

            try:
                print(f"🔍 스크래핑: {current_url} (깊이: {depth}, 도메인: {domain_count + 1})")

                # 콘텐츠 타입별 최적화된 추출
                scrape_result = self.extract_content_by_type(current_url, content_types)

                if scrape_result.get('success') and scrape_result.get('content'):
                    # 중복 검사
                    if not self.duplicate_detector.is_duplicate(scrape_result['content'], current_url):
                        results.append(scrape_result)
                        self.visited_urls.add(current_url)

                        # 도메인 통계 업데이트
                        self.domain_stats[domain] = domain_count + 1
                        total_scraped += 1

                        print(f"✅ 성공: {len(scrape_result['content'])}자 추출")

                        # 링크 수집 (깊이 제한 내에서)
                        if depth < self.config.get('max_depth', 3):
                            new_links = self.extract_relevant_links(
                                current_url,
                                scrape_result.get('html', ''),
                                content_types
                            )

                            for link in new_links:
                                if link not in self.visited_urls:
                                    url_queue.append((link, depth + 1))
                    else:
                        print("⚠️ 중복 콘텐츠 감지됨")
                        # 중복이어도 방문 기록에 추가
                        self.visited_urls.add(current_url)
                        self.domain_stats[domain] = domain_count + 1
                else:
                    print(f"❌ 콘텐츠 추출 실패: {scrape_result.get('error', 'Unknown error')}")
                    # 실패해도 방문 기록에 추가 (무한 재시도 방지)
                    self.visited_urls.add(current_url)

                # 지연
                time.sleep(self.config.get('delay_between_requests', 1))

            except Exception as e:
                print(f"❌ 오류: {current_url} - {str(e)}")
                # 오류 발생해도 방문 기록에 추가
                self.visited_urls.add(current_url)
                continue

        print(f"\n🎉 스크래핑 완료: 총 {total_scraped}개 페이지 수집")
        print(f"📊 도메인별 통계: {dict(self.domain_stats)}")
        print(f"📄 유효 콘텐츠: {len(results)}개")

        # 필터링 통계
        total_visited = len(self.visited_urls)
        filtered_count = total_visited - len(results)
        if filtered_count > 0:
            print(f"🚫 필터링된 페이지: {filtered_count}개")
            print(f"   - 중복 콘텐츠, 광고, 로그인 페이지, 에러 페이지 등")

        print(f"✅ 최종 성공률: {len(results)}/{total_visited} = {(len(results) / max(total_visited, 1) * 100):.1f}%")

        return results

    def extract_content_by_type(self, url, content_types):
        """콘텐츠 타입별 최적화된 추출 - 고급 필터링 추가"""

        result = {
            'url': url,
            'title': '',
            'content': '',
            'content_type': 'unknown',
            'success': False,
            'method': 'none',
            'metadata': {},
            'html': ''
        }

        try:
            # 🚫 URL 사전 필터링 (쓸모없는 링크 차단)
            if self.is_unwanted_url(url):
                result['error'] = "필터링된 URL (광고/후원/만료 등)"
                return result

            # 기본 페이지 가져오기
            response = self.session.get(url, timeout=15)

            # 🚫 HTTP 상태 코드 확인
            if response.status_code == 404:
                result['error'] = "404 Not Found - 만료된 링크"
                return result
            elif response.status_code >= 400:
                result['error'] = f"HTTP {response.status_code} 오류"
                return result

            response.raise_for_status()

            html_content = response.text
            result['html'] = html_content

            # 🚫 로그인 필요 페이지 감지
            if self.is_login_required_page(html_content):
                result['error'] = "로그인이 필요한 페이지"
                return result

            # 🚫 에러 페이지 감지
            if self.is_error_page(html_content):
                result['error'] = "에러 페이지 감지"
                return result

            soup = BeautifulSoup(html_content, 'html.parser')

            # 콘텐츠 타입 감지
            detected_type = self.detect_content_type(soup, url)
            result['content_type'] = detected_type

            # 타입별 최적화된 추출
            if content_types == 'all' or detected_type in content_types:
                content_data = self.extract_by_detected_type(soup, detected_type, url)

                if content_data and content_data.get('content'):
                    # 🚫 콘텐츠 길이 확인
                    content_text = content_data.get('content', '')
                    if len(content_text.strip()) < 200:
                        result['error'] = "콘텐츠가 너무 짧음 (200자 미만)"
                        return result

                    # 🚫 쓸모없는 콘텐츠 감지
                    if self.is_useless_content(content_text):
                        result['error'] = "쓸모없는 콘텐츠 (광고/스팸 등)"
                        return result

                    result.update(content_data)
                    result['success'] = True
                    result['method'] = f'{detected_type}_optimized_filtered'
                else:
                    result['error'] = "유효한 콘텐츠 추출 실패"
            else:
                result['error'] = f"콘텐츠 타입 불일치: {detected_type}"

            return result

        except Exception as e:
            result['error'] = str(e)
            return result

    def is_unwanted_url(self, url):
        """🚫 쓸모없는 URL 패턴 감지"""
        url_lower = url.lower()

        # 광고/후원/상업 관련
        unwanted_patterns = [
            'paypal', 'donate', 'donation', 'sponsor', 'patreon',
            'buy-domain', 'domain-for-sale', 'expired-domain', 'domainshop',
            'godaddy', 'namecheap', 'domain.com',
            'advertisement', 'ads', 'adsense', 'doubleclick', 'googlesyndication',
            'affiliate', 'amzn.to', 'amazon.', 'ebay.', 'alibaba.',
            'shop', 'store', 'cart', 'checkout', 'buy-now', 'purchase',
            'casino', 'betting', 'gambling', 'poker', 'slots',
            'viagra', 'cialis', 'pharmacy', 'pills', 'supplement'
        ]

        # 로그인/가입 관련
        auth_patterns = [
            'login', 'signin', 'sign-in', 'log-in',
            'register', 'signup', 'sign-up', 'join',
            'account', 'profile', 'dashboard', 'admin',
            'password', 'forgot', 'reset'
        ]

        # 기술적 페이지
        technical_patterns = [
            'sitemap', 'robots.txt', 'favicon.ico',
            'css', 'js', '.css', '.js',
            'api/', 'ajax/', 'json', 'xml',
            'feed', 'rss', 'atom',
            'trackback', 'pingback'
        ]

        # 소셜 미디어 (외부)
        social_patterns = [
            'facebook.com', 'twitter.com', 'instagram.com',
            'youtube.com', 'linkedin.com', 'pinterest.com',
            'tiktok.com', 'snapchat.com', 'reddit.com'
        ]

        all_patterns = unwanted_patterns + auth_patterns + technical_patterns + social_patterns

        return any(pattern in url_lower for pattern in all_patterns)

    def is_login_required_page(self, html_content):
        """🚫 로그인 필요 페이지 감지"""
        html_lower = html_content.lower()

        login_indicators = [
            'please log in', 'please sign in', 'login required',
            'access denied', 'unauthorized', 'authentication required',
            'member only', 'subscribers only', 'premium content',
            'you must be logged in', 'sign in to continue',
            'this content is protected', 'private content'
        ]

        # 로그인 폼 감지
        login_form_patterns = [
            '<input[^>]*type=["\']password["\']',
            '<form[^>]*login[^>]*>',
            'name=["\']username["\']',
            'name=["\']email["\'].*name=["\']password["\']'
        ]

        # 텍스트 기반 감지
        for indicator in login_indicators:
            if indicator in html_lower:
                return True

        # 폼 기반 감지
        import re
        for pattern in login_form_patterns:
            if re.search(pattern, html_lower, re.DOTALL):
                return True

        return False

    def is_error_page(self, html_content):
        """🚫 에러 페이지 감지"""
        html_lower = html_content.lower()

        error_indicators = [
            '404 not found', 'page not found', 'file not found',
            '403 forbidden', 'access forbidden', 'permission denied',
            '500 internal server error', 'server error',
            'this page doesn\'t exist', 'page has been removed',
            'content not available', 'temporarily unavailable',
            'site under construction', 'coming soon',
            'domain expired', 'domain for sale', 'parked domain'
        ]

        return any(indicator in html_lower for indicator in error_indicators)

    def is_useless_content(self, content):
        """🚫 쓸모없는 콘텐츠 감지"""
        content_lower = content.lower()

        # 너무 짧거나 긴 경우
        if len(content.strip()) < 200:
            return True

        # 광고성 콘텐츠
        spam_patterns = [
            'click here to buy', 'limited time offer', 'act now',
            'buy now', 'order today', 'special discount',
            'make money', 'earn money', 'get rich',
            'weight loss', 'lose weight', 'diet pills',
            'casino', 'poker', 'gambling', 'bet now',
            'viagra', 'cialis', 'enlargement'
        ]

        # 스팸 패턴 개수 확인
        spam_count = sum(1 for pattern in spam_patterns if pattern in content_lower)

        # 3개 이상 스팸 패턴이 있으면 쓸모없는 콘텐츠로 판단
        if spam_count >= 3:
            return True

        # 반복되는 키워드 (SEO 스팸)
        words = content_lower.split()
        if len(words) > 50:
            from collections import Counter
            word_freq = Counter(words)
            # 가장 빈번한 단어가 전체의 10% 이상이면 스팸 의심
            most_common_freq = word_freq.most_common(1)[0][1]
            if most_common_freq / len(words) > 0.1:
                return True

        return False

    def detect_content_type(self, soup, url):
        """웹페이지 콘텐츠 타입 자동 감지"""

        url_lower = url.lower()

        # URL 패턴 기반 감지
        if any(pattern in url_lower for pattern in ['blog', 'diary', 'personal']):
            return 'blog'
        elif any(pattern in url_lower for pattern in ['news', 'article', 'press']):
            return 'news'
        elif any(pattern in url_lower for pattern in ['forum', 'community', 'board']):
            return 'forum'
        elif any(pattern in url_lower for pattern in ['wiki', 'reference', 'encyclopedia']):
            return 'reference'
        elif any(pattern in url_lower for pattern in ['research', 'paper', 'journal', 'academic']):
            return 'academic'

        # HTML 구조 기반 감지
        # 블로그 감지
        if soup.find_all(class_=re.compile(r'post|article|blog', re.I)):
            return 'blog'

        # 뉴스 감지
        if soup.find_all(class_=re.compile(r'news|article|story', re.I)):
            return 'news'

        # 포럼 감지
        if soup.find_all(class_=re.compile(r'forum|thread|reply|comment', re.I)):
            return 'forum'

        # 학술 감지
        if soup.find_all(class_=re.compile(r'abstract|paper|research|journal', re.I)):
            return 'academic'

        # 기본값
        return 'general'

    def extract_by_detected_type(self, soup, content_type, url):
        """감지된 타입별 최적화된 콘텐츠 추출"""

        extractors = {
            'blog': self.extract_blog_content,
            'news': self.extract_news_content,
            'forum': self.extract_forum_content,
            'academic': self.extract_academic_content,
            'reference': self.extract_reference_content,
            'general': self.extract_general_content
        }

        extractor = extractors.get(content_type, self.extract_general_content)
        return extractor(soup, url)

    def extract_blog_content(self, soup, url):
        """블로그 콘텐츠 추출"""
        # 제목 추출
        title = self.extract_title(soup)

        # 본문 추출 - 블로그 특화
        content_selectors = [
            '.post-content', '.entry-content', '.blog-content',
            '.post-body', '.article-content', '.content',
            'article', '.post', '.entry'
        ]

        content = self.extract_content_by_selectors(soup, content_selectors)

        # 메타데이터 추출
        metadata = {
            'author': self.extract_author(soup),
            'date': self.extract_date(soup),
            'tags': self.extract_tags(soup),
            'category': self.extract_category(soup)
        }

        return {
            'title': title,
            'content': f"제목: {title}\n\n{content}",
            'metadata': metadata
        }

    def extract_news_content(self, soup, url):
        """뉴스 콘텐츠 추출"""
        title = self.extract_title(soup)

        # 뉴스 특화 선택자
        content_selectors = [
            '.article-body', '.news-content', '.story-body',
            '.article-text', '.news-text', '.content-area',
            'article', '.story', '.news-article'
        ]

        content = self.extract_content_by_selectors(soup, content_selectors)

        metadata = {
            'author': self.extract_author(soup),
            'date': self.extract_date(soup),
            'source': self.extract_news_source(soup),
            'category': self.extract_category(soup)
        }

        return {
            'title': title,
            'content': f"뉴스 제목: {title}\n\n{content}",
            'metadata': metadata
        }

    def extract_forum_content(self, soup, url):
        """포럼 콘텐츠 추출"""
        title = self.extract_title(soup)

        # 포럼 특화 - 메인 포스트와 댓글 분리
        main_content = self.extract_content_by_selectors(soup, [
            '.thread-content', '.post-content', '.forum-post',
            '.main-post', '.original-post'
        ])

        # 댓글들 추출
        comments = self.extract_forum_comments(soup)

        full_content = f"제목: {title}\n\n본문:\n{main_content}"
        if comments:
            full_content += f"\n\n댓글들:\n{comments}"

        return {
            'title': title,
            'content': full_content,
            'metadata': {
                'forum': True,
                'comments_count': len(comments.split('\n---')) if comments else 0
            }
        }

    def extract_academic_content(self, soup, url):
        """학술 콘텐츠 추출"""
        title = self.extract_title(soup)

        # 학술 논문 특화
        abstract = self.extract_content_by_selectors(soup, [
            '.abstract', '.summary', '.abstract-content'
        ])

        main_content = self.extract_content_by_selectors(soup, [
            '.article-body', '.paper-content', '.full-text',
            '.content', 'article', '.paper-body'
        ])

        content = f"제목: {title}\n\n"
        if abstract:
            content += f"초록: {abstract}\n\n"
        content += f"본문: {main_content}"

        return {
            'title': title,
            'content': content,
            'metadata': {
                'academic': True,
                'has_abstract': bool(abstract)
            }
        }

    def extract_reference_content(self, soup, url):
        """참조 자료 (위키 등) 콘텐츠 추출"""
        title = self.extract_title(soup)

        content_selectors = [
            '.mw-parser-output', '.wiki-content', '.content',
            '.reference-content', 'article', '.main-content'
        ]

        content = self.extract_content_by_selectors(soup, content_selectors)

        return {
            'title': title,
            'content': f"참조 자료: {title}\n\n{content}",
            'metadata': {'reference': True}
        }

    def extract_general_content(self, soup, url):
        """일반 웹페이지 콘텐츠 추출"""
        title = self.extract_title(soup)

        # 광고 및 불필요한 요소 제거
        self.remove_unwanted_elements(soup)

        # 일반적인 콘텐츠 선택자들
        content_selectors = [
            'main', 'article', '.content', '.main-content',
            '.page-content', '.entry-content', '.post-content',
            '#content', '#main', '.container'
        ]

        content = self.extract_content_by_selectors(soup, content_selectors)

        # 최후의 수단: body에서 직접 추출
        if not content or len(content) < 200:
            content = self.extract_from_body(soup)

        return {
            'title': title,
            'content': f"제목: {title}\n\n{content}",
            'metadata': {'general': True}
        }

    def extract_content_by_selectors(self, soup, selectors):
        """선택자들로 콘텐츠 추출"""
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                for element in elements:
                    text = element.get_text(strip=True)
                    if len(text) > 100:  # 충분한 콘텐츠가 있는 경우
                        return self.clean_text(text)
        return ""

    def extract_title(self, soup):
        """제목 추출"""
        # 여러 방법으로 제목 추출 시도
        title_selectors = [
            'h1', '.title', '.post-title', '.article-title',
            '.entry-title', '.page-title', 'title'
        ]

        for selector in title_selectors:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 5:
                    return title

        return "제목 없음"

    def extract_author(self, soup):
        """작성자 추출"""
        author_selectors = [
            '.author', '.byline', '.writer', '.post-author',
            '.article-author', '[rel="author"]'
        ]

        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None

    def extract_date(self, soup):
        """날짜 추출"""
        date_selectors = [
            '.date', '.published', '.post-date', '.article-date',
            'time', '.timestamp', '.created'
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                # datetime 속성 확인
                if element.get('datetime'):
                    return element.get('datetime')
                return element.get_text(strip=True)

        return None

    def extract_tags(self, soup):
        """태그 추출"""
        tag_selectors = [
            '.tags', '.tag', '.post-tags', '.article-tags',
            '.categories', '.labels'
        ]

        tags = []
        for selector in tag_selectors:
            elements = soup.select(selector)
            for element in elements:
                tags.append(element.get_text(strip=True))

        return tags if tags else None

    def extract_category(self, soup):
        """카테고리 추출"""
        category_selectors = [
            '.category', '.post-category', '.article-category',
            '.breadcrumb', '.nav-category'
        ]

        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None

    def extract_news_source(self, soup):
        """뉴스 출처 추출"""
        source_selectors = [
            '.source', '.news-source', '.publisher',
            '.publication', '.media-source'
        ]

        for selector in source_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None

    def extract_forum_comments(self, soup):
        """포럼 댓글 추출"""
        comment_selectors = [
            '.comment', '.reply', '.forum-reply',
            '.thread-reply', '.comment-content'
        ]

        comments = []
        for selector in comment_selectors:
            elements = soup.select(selector)
            for element in elements:
                comment_text = element.get_text(strip=True)
                if len(comment_text) > 20:
                    comments.append(comment_text)

        return '\n---\n'.join(comments) if comments else ""

    def extract_from_body(self, soup):
        """body에서 직접 텍스트 추출"""
        if soup.body:
            return self.clean_text(soup.body.get_text())
        return ""

    def remove_unwanted_elements(self, soup):
        """불필요한 요소들 제거"""
        unwanted_selectors = [
            'script', 'style', 'nav', 'footer', 'header',
            '.advertisement', '.ad', '.ads', '.sidebar',
            '.menu', '.navigation', '.popup', '.modal',
            '.social-share', '.related-posts', '.comments'
        ]

        for selector in unwanted_selectors:
            for element in soup.select(selector):
                element.decompose()

    def clean_text(self, text):
        """텍스트 정리"""
        # 여러 줄바꿈을 하나로
        text = re.sub(r'\n+', '\n', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_relevant_links(self, base_url, html_content, content_types):
        """관련 링크 추출 - 고급 필터링 적용"""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        base_domain = urlparse(base_url).netloc

        print(f"🔍 {base_url}에서 링크 추출 중...")

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # 🚫 기본 필터링
            if parsed.scheme not in ['http', 'https']:
                continue

            # 🚫 다른 도메인 제외 (같은 사이트 내에서만)
            if parsed.netloc != base_domain:
                continue

            # 🚫 파일 확장자 제외
            if any(parsed.path.lower().endswith(ext) for ext in [
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.zip', '.rar', '.jpg', '.jpeg', '.png', '.gif', '.svg',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.css', '.js'
            ]):
                continue

            # 🚫 쓸모없는 URL 패턴 제외
            if self.is_unwanted_url(full_url):
                print(f"   ❌ 필터링: {full_url[:50]}... (쓸모없는 패턴)")
                continue

            # 🚫 링크 텍스트 기반 필터링
            link_text = a_tag.get_text().strip().lower()
            if self.is_unwanted_link_text(link_text):
                print(f"   ❌ 필터링: {full_url[:50]}... (쓸모없는 링크 텍스트)")
                continue

            # 🚫 중복 URL 제거
            if full_url in links:
                continue

            # ✅ 콘텐츠 타입별 링크 관련성 확인
            if self.is_relevant_link(full_url, a_tag, content_types):
                links.append(full_url)
                print(f"   ✅ 추가: {full_url[:50]}...")

        print(f"📊 총 {len(links)}개 유효 링크 발견")
        return links[:20]  # 링크 수 제한

    def is_unwanted_link_text(self, link_text):
        """🚫 쓸모없는 링크 텍스트 감지"""
        if not link_text or len(link_text) < 2:
            return True

        unwanted_texts = [
            # 광고/상업
            'buy now', 'shop now', 'order now', 'purchase',
            'add to cart', 'checkout', 'buy', 'sale', 'discount',
            'sponsor', 'donate', 'donation', 'paypal',

            # 로그인/가입
            'sign in', 'log in', 'login', 'register', 'sign up',
            'join now', 'member', 'account', 'profile',

            # 소셜/공유
            'share', 'like', 'follow', 'subscribe', 'tweet',
            'facebook', 'twitter', 'instagram', 'youtube',

            # 기술적
            'rss', 'feed', 'sitemap', 'contact', 'about us',
            'privacy policy', 'terms', 'cookie policy',

            # 네비게이션 (너무 일반적)
            'home', 'back', 'next', 'previous', 'top',
            'menu', 'search', 'more', 'all', 'list',

            # 광고성
            'click here', 'read more', 'learn more', 'find out',
            'discover', 'explore', 'get started', 'try now'
        ]

        # 정확한 매치
        if link_text in unwanted_texts:
            return True

        # 부분 매치 (짧은 텍스트의 경우)
        if len(link_text) < 10:
            for unwanted in unwanted_texts:
                if unwanted in link_text:
                    return True

        # 숫자만 있는 링크 (페이지네이션)
        if link_text.isdigit():
            return True

        # 한 글자 링크
        if len(link_text) == 1:
            return True

        return False

    def is_relevant_link(self, url, link_element, content_types):
        """링크 관련성 판단 - 개선된 버전"""
        if content_types == 'all':
            return True

        url_lower = url.lower()
        link_text = link_element.get_text().lower()

        # 콘텐츠 타입별 관련성 키워드 (확장)
        relevance_keywords = {
            'academic': [
                'paper', 'research', 'study', 'journal', 'publication',
                'article', 'thesis', 'dissertation', 'academic', 'scholar',
                'science', 'analysis', 'data', 'report', 'findings'
            ],
            'news': [
                'news', 'article', 'story', 'report', 'breaking',
                'update', 'latest', 'current', 'today', 'press',
                'announcement', 'release', 'coverage', 'headline'
            ],
            'blog': [
                'blog', 'post', 'diary', 'personal', 'opinion',
                'thoughts', 'experience', 'review', 'commentary',
                'reflection', 'insight', 'perspective', 'story'
            ],
            'forum': [
                'forum', 'discussion', 'thread', 'community', 'talk',
                'debate', 'conversation', 'comment', 'reply',
                'question', 'answer', 'help', 'support'
            ],
            'reference': [
                'wiki', 'reference', 'guide', 'manual', 'handbook',
                'documentation', 'tutorial', 'howto', 'faq',
                'encyclopedia', 'glossary', 'definition'
            ]
        }

        if isinstance(content_types, str):
            content_types = [content_types]

        for content_type in content_types:
            keywords = relevance_keywords.get(content_type, [])

            # URL이나 링크 텍스트에 관련 키워드가 있는지 확인
            for keyword in keywords:
                if keyword in url_lower or keyword in link_text:
                    return True

        # 기본적으로 허용 (너무 제한적이지 않게)
        return True


class ImprovedEconomicDataProcessor:
    """경제 데이터 처리 개선 클래스"""

    def __init__(self):
        self.country_mapping = {
            '프랑스': 'France', '독일': 'Germany', '영국': 'UK', '미국': 'USA',
            '일본': 'Japan', '중국': 'China', '한국': 'S. Korea', '대한민국': 'S. Korea',
            '이탈리아': 'Italy', '스페인': 'Spain', '캐나다': 'Canada', '호주': 'Australia',
            '브라질': 'Brazil', '인도': 'India', '러시아': 'Russia', '멕시코': 'Mexico'
        }

    def process_maddison_gdp_data(self, file_path):
        """Maddison GDP 데이터 특화 처리"""
        try:
            # 파일 확장자에 따른 엔진 선택
            file_ext = Path(file_path).suffix.lower()

            if file_ext == '.xls':
                if not XLRD_AVAILABLE:
                    print(f"⚠️ xlrd 패키지가 없어서 .xls 파일을 처리할 수 없습니다")
                    return []
                engine = 'xlrd'
            else:
                engine = 'openpyxl'

            # GDP 시트 읽기
            try:
                df_gdp = pd.read_excel(file_path, sheet_name='GDP', header=None, engine=engine)
            except Exception as e:
                print(f"⚠️ {engine} 엔진으로 GDP 시트 읽기 실패: {e}")
                # 다른 엔진으로 시도
                alternative_engine = 'openpyxl' if engine == 'xlrd' else 'xlrd'
                try:
                    df_gdp = pd.read_excel(file_path, sheet_name='GDP', header=None, engine=alternative_engine)
                    print(f"✅ {alternative_engine} 엔진으로 성공")
                except Exception as e2:
                    print(f"❌ 모든 엔진 실패: {e2}")
                    return []

            # 국가 헤더 행 (3번째 행, 0-based index = 2)
            if len(df_gdp) <= 2:
                print("⚠️ GDP 시트 데이터가 부족합니다")
                return []

            country_row = df_gdp.iloc[2].fillna('')

            # 연도별 데이터 추출 (4번째 행부터)
            year_data = df_gdp.iloc[3:].reset_index(drop=True)

            documents = []
            processed_countries = 0

            # 각 국가별로 처리
            for col_idx, country in enumerate(country_row):
                if not country or len(str(country).strip()) < 2:
                    continue

                country_name = str(country).strip()

                # 국가별 GDP 데이터 추출
                if col_idx >= len(year_data.columns):
                    continue

                gdp_series = year_data.iloc[:, col_idx]
                year_series = year_data.iloc[:, 0]  # 첫 번째 열은 연도

                # 유효한 데이터만 필터링
                valid_data = []
                for year, gdp in zip(year_series, gdp_series):
                    if pd.notna(year) and pd.notna(gdp) and year != '' and gdp != '':
                        try:
                            year_int = int(float(year))
                            gdp_float = float(gdp)
                            if 1800 <= year_int <= 2030 and gdp_float > 0:
                                valid_data.append((year_int, gdp_float))
                        except (ValueError, TypeError):
                            continue

                if valid_data and len(valid_data) > 5:  # 최소 5개 이상의 유효 데이터
                    content = self.create_searchable_gdp_content(country_name, valid_data)

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'source': str(file_path),
                            'type': 'economic_data',
                            'country': country_name,
                            'data_type': 'GDP',
                            'years_covered': f"{min(d[0] for d in valid_data)}-{max(d[0] for d in valid_data)}"
                        }
                    ))
                    processed_countries += 1

            print(f"✅ {processed_countries}개 국가의 GDP 데이터 처리 완료")
            return documents

        except Exception as e:
            print(f"❌ Maddison 데이터 처리 오류: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return []

    def create_searchable_gdp_content(self, country_name, year_gdp_data):
        """검색하기 쉬운 GDP 콘텐츠 생성"""

        # 한국어 국가명 추가
        korean_name = None
        for ko, en in self.country_mapping.items():
            if en.lower() == country_name.lower():
                korean_name = ko
                break

        content = f"=== {country_name} GDP 데이터 ===\n"
        if korean_name:
            content += f"국가명: {korean_name} ({country_name})\n"

        content += "데이터 출처: Angus Maddison Historical Statistics of the World Economy\n"
        content += "단위: 1990년 국제 달러 (1990 International Geary-Khamis dollars)\n\n"

        # 연도별 데이터를 검색하기 쉽게 포맷
        content += "연도별 GDP 데이터:\n"

        # 최근 30년 데이터를 우선적으로 표시
        recent_data = [(year, gdp) for year, gdp in year_gdp_data if year >= 1980]

        # 최근 데이터 먼저
        for year, gdp in sorted(recent_data, reverse=True):
            content += f"- {year}년: {gdp:,.0f}\n"
            content += f"- {country_name} {year} GDP: {gdp:,.0f}\n"
            if korean_name:
                content += f"- {korean_name} {year}년 GDP: {gdp:,.0f}\n"

        # 특별히 중요한 연도들 강조
        important_years = [1990, 1995, 2000, 2005, 2008]
        content += "\n주요 연도 GDP:\n"

        for year, gdp in year_gdp_data:
            if year in important_years:
                content += f"🔸 {year}년 {country_name} GDP: {gdp:,.0f} (1990 Int. $)\n"
                if korean_name:
                    content += f"🔸 {year}년 {korean_name} GDP: {gdp:,.0f}\n"

        # 2000년 특별 강조
        y2000_data = [gdp for year, gdp in year_gdp_data if year == 2000]
        if y2000_data:
            content += f"\n*** 2000년 {country_name} GDP ***\n"
            content += f"값: {y2000_data[0]:,.0f} (1990년 국제 달러)\n"
            if korean_name:
                content += f"{korean_name} 2000년 GDP: {y2000_data[0]:,.0f}\n"

        return content


class MilitaryAnalysisTemplates:
    """군사·전력 분석 전문 템플릿"""

    @staticmethod
    def get_military_analysis_template():
        """군사·전력·국가모의전 전문 분석 템플릿"""
        return ChatPromptTemplate.from_template("""
당신은 국가모의전/군사사(무기·전력·산업동원) 전문 분석가입니다. 정량 근거와 신뢰 가능한 출처를 최우선으로 하며, 과장·영웅서사를 배제하고 **건조하고 사실 중심**으로 답변합니다. 사용자가 틀리면 단호하게 바로잡습니다.

**참고 문서:**
{context}

**사용자 질문:** {question}

**검색 쿼리 생성 규칙:**
- 핵심 엔티티(국가/연도/전구/부대/장비 모델)를 모두 포함
- 동의어/약어를 OR로 확장: "order of battle" OR OOB OR 편제, artillery OR 야포, armored OR 장갑 등
- 기간 필터: 해당 연도 ± 1~2년, 사건은 정확 날짜 포함

**출처 우선순위:** 
1차 – 정부/군/연감, Jane's, IISS Military Balance, SIPRI, 학술서(전사)
2차 – 신뢰 가능한 싱크탱크/저널
3차 – 언론·블로그는 보조 검증용(단독 인용 금지)

**응답 원칙(반드시 준수):**
1. **사실 vs 가정 분리:** 역사적 사실과 What-if(대안사)를 명확히 구분
2. **정량 우선:** 병력·장비 수량, 구경, 사거리, 속도, 생산량/기간 등 숫자 제시
3. **절대 날짜:** "최근/요즘" 금지. YYYY-MM-DD로 표기
4. **현실성 검증:** 산업·훈련·정비·보급 제약을 근거로 실행 가능성 평가
5. **인도주의/규범:** 민간인 대상 폭력·학살 정당화 금지. ROE/IHL 전제 명시
6. **불확실성 표기:** 수치가 엇갈리면 범위·신뢰도 표시(예: "3,000~3,500명, 신뢰도 중")
7. **반박 허용:** 사용자 주장에 오류가 있으면 근거로 반박
8. **톤:** 간결·건조·현실적. 감상·추측형 수사 금지

**출력 형식 (마크다운):**

# 요약
핵심 결론과 숫자 2~4개 (한 문단)

## 핵심 표
- 전력 편성(OOB) 표: 부대/병력/주요 장비/배치/날짜
- 무기 제원 표(필요시): 모델/구경/사거리/속도/장갑/비고
- 생산·동원 요약 표: 품목/월간 생산량/리드타임/병목/비고

## 상세 분석

### 1) 사실 (Facts)
검증된 사실만 나열. 각 문장 끝에 [S1], [S2] 식의 문헌 표기

### 2) 가정/시나리오 (What-if)
가정 조건 → 단계별 전개 → 성공/리스크 → 분기점

### 3) 실행 가능성 (현실성 체크)
산업·인력·훈련·정비·보급 관점에서 제약과 병목

### 4) 손실/대가 추정
인원·장비 손실 범위, 정치·외교·경제 비용(정성/정량)

### 5) 불확실성/한계
데이터 공백, 상충 출처, 추가로 필요한 1차 사료

## 출처
[S1] 저자/기관, 제목, 연도(페이지/표/도)
[S2] ...

**검증 체크리스트:**
- 대상 연도·전구·부대 정확히 특정?
- 모든 숫자에 근거 표기?
- 사실/가정 분리 완료?
- 로지스틱스(연료·탄약·수송·정비) 고려?
- 인도주의 제약과 민간 피해 최소화 전제 명시?
- 불일치 수치에 범위·신뢰도 기재?

**금지사항:**
- 출처 불명 수치의 단정 서술
- 음모론/선동 인용
- 영화·게임 설정은 픽션으로 명시
- 모호한 시간 표현

**답변:**""")


class UltimateMilitaryRAGv6:
    """군사·전력 분석 전문 RAG 시스템 v6.0"""

    def __init__(self, llm_model="qwen2.5:7b"):  # 기본 모델을 qwen2.5:7b로 변경
        print("🚀 군사·전력 분석 전문 RAG 시스템 v6.0 초기화...")

        # 필수 속성들을 먼저 초기화 (예외 발생해도 최소한의 상태 유지)
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.embeddings = None
        self.llm = None

        try:
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
                raise Exception("Ollama 서버 연결 실패")

            # 로컬 임베딩 모델 초기화
            print("🔄 로컬 임베딩 모델 로딩...")
            device = self.force_gpu_check()

            try:
                model_kwargs = {'device': device}
                encode_kwargs = {'normalize_embeddings': True, 'batch_size': 16}

                self.embeddings = HuggingFaceEmbeddings(
                    model_name="intfloat/multilingual-e5-large",
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
                print("✅ 메인 임베딩 모델 로드 완료")

            except Exception as e:
                # 백업 모델(384차원)은 메인 모델(1024차원)로 만든 벡터 DB를 검색할 수 없다.
                # 기존 DB가 있으면 조용히 바꾸지 않고 원인과 해결 방법을 알린다.
                existing = self.existing_vectorstore_paths()
                if existing:
                    paths = ", ".join(str(path) for path in existing)
                    raise RuntimeError(
                        f"메인 임베딩 모델(intfloat/multilingual-e5-large, 1024차원)을 불러오지 못했습니다: {e}\n"
                        f"기존 벡터 DB({paths})는 이 모델로 만들어져 있어, 백업 모델"
                        f"(all-MiniLM-L6-v2, 384차원)로는 차원이 달라 검색할 수 없으므로 백업 모델로 바꾸지 않습니다.\n"
                        "해결: ① 메인 모델 로드 문제(인터넷 연결·Hugging Face 캐시·디스크 공간·"
                        "sentence-transformers 설치)를 해결한 뒤 다시 실행하거나, "
                        "② 백업 모델로 새로 구축하려면 위 DB 폴더를 다른 곳으로 옮기거나 지운 뒤 다시 실행하세요."
                    ) from e
                print(f"⚠️ 메인 모델 실패, 백업 모델 사용: {e}")
                try:
                    self.embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    print("⚠️ 백업 임베딩 모델(all-MiniLM-L6-v2, 384차원)로 대체했습니다. 검색 품질이 낮아질 수 있고, "
                          "이 상태로 만든 벡터 DB는 메인 모델(1024차원)이 다시 로드되면 사용할 수 없어 새로 구축해야 합니다.")
                except Exception as e2:
                    print(f"❌ 모든 임베딩 모델 실패: {e2}")
                    raise Exception(f"임베딩 모델 로드 실패: {e2}")

            # Ollama LLM 초기화 - 최신 버전 대응
            print(f"🤖 {llm_model} 모델 연결 중...")
            base_url = ollama_base_url()

            if OLLAMA_NEW:
                # 최신 langchain-ollama 패키지
                self.llm = Ollama(
                    model=llm_model,
                    temperature=0.1,
                    num_ctx=8192,
                    base_url=base_url
                )
                print("✅ 최신 langchain-ollama 사용")
            else:
                # 구버전 langchain-community
                self.llm = Ollama(
                    model=llm_model,
                    temperature=0.1,
                    num_ctx=8192,
                    base_url=base_url
                )
                print("⚠️ 구버전 ollama 사용 - 업그레이드 권장: pip install -U langchain-ollama")

            # 전문 컴포넌트 초기화
            self.personal_scraper = PersonalDataWebScraper()
            self.economic_processor = ImprovedEconomicDataProcessor()
            self.military_templates = MilitaryAnalysisTemplates()

            # 스크래핑 기본 설정
            self.personal_scraper.configure_scraping_limits(
                max_pages_total=1000,
                max_pages_per_domain=100,
                max_depth=3,
                delay_between_requests=1
            )

            # 스크래핑 콘텐츠 타입 초기화
            self.scraping_content_types = "all"

            # 군사 전문 프롬프트 설정
            self.current_template = self.military_templates.get_military_analysis_template()

            # 기존 벡터스토어 로드 시도
            self.load_vectorstore()

            print("✅ v6.0 군사·전력 분석 시스템 초기화 완료!")

        except Exception as e:
            print(f"❌ 초기화 중 오류 발생: {e}")
            # 기본값으로 설정하여 최소한의 기능 유지
            if not hasattr(self, 'personal_scraper') or self.personal_scraper is None:
                self.personal_scraper = PersonalDataWebScraper()
            if not hasattr(self, 'economic_processor') or self.economic_processor is None:
                self.economic_processor = ImprovedEconomicDataProcessor()
            if not hasattr(self, 'military_templates') or self.military_templates is None:
                self.military_templates = MilitaryAnalysisTemplates()
            if not hasattr(self, 'current_template') or self.current_template is None:
                self.current_template = self.military_templates.get_military_analysis_template()
            if not hasattr(self, 'scraping_content_types'):
                self.scraping_content_types = "all"

            # 오류를 다시 발생시켜 상위에서 처리하도록 함
            raise e

    def existing_vectorstore_paths(self):
        """기본 경로와 대안 경로(임시 폴더) 중 이미 존재하는 벡터 DB 폴더 목록"""
        import tempfile
        candidates = [self.db_path / "chroma_db_v6_military",
                      Path(tempfile.gettempdir()) / "chroma_db_v6_military"]
        return [path for path in candidates if path.exists()]

    def test_ollama_connection(self):
        """Ollama 서버 연결 테스트"""
        try:
            response = requests.get(f"{ollama_base_url()}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False

    def force_gpu_check(self):
        """GPU 강제 설정 시도"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                return 'cuda'
            else:
                return 'cpu'
        except Exception:
            return 'cpu'

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
                        # 경제 데이터 파일 특별 처리
                        if any(keyword in file_path.name.lower() for keyword in ['vertical', 'maddison', 'gdp']):
                            econ_docs = self.load_economic_excel_file(file_path)
                            documents.extend(econ_docs)
                        else:
                            docs = file_processors[file_ext](file_path)
                            documents.extend(docs)
                    except Exception as e:
                        print(f"❌ {file_path.name} 처리 실패: {e}")

        print(f"📊 총 {len(documents)}개 문서 로드 완료")
        return documents

    def load_economic_excel_file(self, file_path):
        """경제 데이터 전용 엑셀 파일 로더"""
        documents = []

        try:
            # 파일 확장자에 따른 엔진 선택
            file_ext = file_path.suffix.lower()

            if file_ext == '.xls':
                if not XLRD_AVAILABLE:
                    print(f"⚠️ .xls 파일 처리를 위해 xlrd가 필요합니다: {file_path.name}")
                    return self.load_excel_file(file_path)  # 기본 방식으로 폴백

                try:
                    xl_file = pd.ExcelFile(file_path, engine='xlrd')
                except Exception:
                    print(f"⚠️ xlrd 엔진 실패, 기본 방식 사용: {file_path.name}")
                    return self.load_excel_file(file_path)
            else:
                try:
                    xl_file = pd.ExcelFile(file_path, engine='openpyxl')
                except Exception:
                    print(f"⚠️ openpyxl 엔진 실패, 기본 방식 사용: {file_path.name}")
                    return self.load_excel_file(file_path)

            # GDP 시트 확인
            if 'GDP' in xl_file.sheet_names:
                print(f"📊 경제 데이터셋 감지: {file_path.name}")
                try:
                    gdp_docs = self.economic_processor.process_maddison_gdp_data(file_path)
                    documents.extend(gdp_docs)
                    print(f"✅ {len(gdp_docs)}개 국가 GDP 데이터 처리 완료")
                except Exception as e:
                    print(f"⚠️ GDP 데이터 처리 실패: {e}")
                    return self.load_excel_file(file_path)

            # Per Capita GDP도 처리
            if 'PerCapita GDP' in xl_file.sheet_names:
                try:
                    percapita_docs = self.process_percapita_gdp(file_path)
                    documents.extend(percapita_docs)
                except Exception as e:
                    print(f"⚠️ 1인당 GDP 처리 실패: {e}")

            # 문서가 없으면 기본 방식으로 처리
            if not documents:
                print(f"📄 기본 Excel 처리 방식 사용: {file_path.name}")
                return self.load_excel_file(file_path)

            return documents

        except Exception as e:
            print(f"❌ 경제 데이터 로드 실패: {e}")
            print(f"📄 기본 Excel 처리 방식으로 폴백: {file_path.name}")
            return self.load_excel_file(file_path)

    def process_percapita_gdp(self, file_path):
        """1인당 GDP 데이터 처리"""
        try:
            df = pd.read_excel(file_path, sheet_name='PerCapita GDP', header=None)
            country_row = df.iloc[2].fillna('')
            year_data = df.iloc[3:].reset_index(drop=True)

            documents = []

            for col_idx, country in enumerate(country_row):
                if not country or len(str(country).strip()) < 2:
                    continue

                country_name = str(country).strip()
                gdp_series = year_data.iloc[:, col_idx]
                year_series = year_data.iloc[:, 0]

                valid_data = []
                for year, gdp in zip(year_series, gdp_series):
                    if pd.notna(year) and pd.notna(gdp):
                        try:
                            year_int = int(float(year))
                            gdp_float = float(gdp)
                            if 1800 <= year_int <= 2030 and gdp_float > 0:
                                valid_data.append((year_int, gdp_float))
                        except:
                            continue

                if valid_data:
                    # 한국어 국가명 매핑
                    korean_name = None
                    for ko, en in self.economic_processor.country_mapping.items():
                        if en.lower() == country_name.lower():
                            korean_name = ko
                            break

                    content = f"=== {country_name} 1인당 GDP 데이터 ===\n"
                    if korean_name:
                        content += f"국가명: {korean_name} ({country_name})\n"
                    content += "단위: 1990년 국제 달러\n\n"

                    for year, gdp in sorted(valid_data, reverse=True):
                        if year >= 1990:
                            content += f"{year}년: ${gdp:,.2f}\n"
                            if korean_name:
                                content += f"{korean_name} {year}년 1인당 GDP: ${gdp:,.2f}\n"

                    documents.append(Document(
                        page_content=content,
                        metadata={
                            'source': str(file_path),
                            'type': 'economic_data',
                            'country': country_name,
                            'data_type': 'GDP_per_capita'
                        }
                    ))

            return documents

        except Exception as e:
            print(f"1인당 GDP 처리 오류: {e}")
            return []

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
        if LANGCHAIN_PDF_AVAILABLE:
            try:
                loader = PDFLoader(str(file_path))
                return loader.load()
            except Exception:
                pass

        # PyPDF2 백업
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
                    metadata={"source": str(file_path), "type": "pdf"}
                ))
                return documents

        except Exception:
            pass

        return [Document(
            page_content=f"PDF 파일 처리 실패: {file_path.name}",
            metadata={"source": str(file_path), "type": "pdf_error"}
        )]

    def load_csv_file(self, file_path):
        """CSV 파일 로드"""
        csv_configs = [
            {'encoding': 'utf-8', 'sep': ',', 'low_memory': False},
            {'encoding': 'cp949', 'sep': ',', 'low_memory': False},
            {'encoding': 'euc-kr', 'sep': ',', 'low_memory': False}
        ]

        df = None
        for config in csv_configs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = pd.read_csv(file_path, **config)
                if len(df) > 0:
                    break
            except Exception:
                continue

        if df is None:
            return [Document(
                page_content=f"CSV 파일 처리 실패: {file_path.name}",
                metadata={"source": str(file_path), "type": "csv_error"}
            )]

        # DataFrame 처리
        content = f"CSV 파일: {file_path.name}\n"
        content += f"행 수: {len(df):,}개, 열 수: {len(df.columns)}개\n\n"

        # 열 이름
        content += f"열 이름: {', '.join(str(col) for col in df.columns[:10])}\n\n"

        # 샘플 데이터
        content += "샘플 데이터:\n"
        content += str(df.head(5))

        return [Document(
            page_content=content,
            metadata={"source": str(file_path), "type": "csv"}
        )]

    def load_excel_file(self, file_path):
        """Excel 파일 로드 - 엔진 자동 선택"""
        documents = []
        file_ext = file_path.suffix.lower()

        # 엔진 우선순위 설정
        engines_to_try = []
        if file_ext == '.xls':
            if XLRD_AVAILABLE:
                engines_to_try = ['xlrd', 'openpyxl']
            else:
                engines_to_try = ['openpyxl']
                print(f"⚠️ .xls 파일 {file_path.name}에 xlrd가 권장되지만 openpyxl로 시도합니다")
        else:  # .xlsx
            engines_to_try = ['openpyxl', 'xlrd'] if XLRD_AVAILABLE else ['openpyxl']

        xl_file = None
        successful_engine = None

        # 엔진별로 시도
        for engine in engines_to_try:
            try:
                xl_file = pd.ExcelFile(file_path, engine=engine)
                successful_engine = engine
                print(f"✅ {file_path.name} - {engine} 엔진 성공")
                break
            except Exception as e:
                print(f"⚠️ {file_path.name} - {engine} 엔진 실패: {e}")
                continue

        if xl_file is None:
            return [Document(
                page_content=f"Excel 파일 처리 실패: {file_path.name}\n모든 엔진 실패",
                metadata={"source": str(file_path), "type": "excel_error"}
            )]

        # 시트 처리
        for sheet_name in xl_file.sheet_names[:10]:  # 최대 10개 시트
            try:
                # 선택된 엔진으로 시트 읽기
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine=successful_engine)

                if len(df) == 0:
                    continue

                content = f"Excel 파일: {file_path.name}, 시트: {sheet_name}\n"
                content += f"엔진: {successful_engine}\n"
                content += f"행 수: {len(df):,}개, 열 수: {len(df.columns)}개\n\n"

                # 열 이름 (안전하게 처리)
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

                # 샘플 데이터 (안전하게 처리)
                try:
                    sample_df = df.head(5).copy()
                    # 모든 열을 문자열로 변환 (안전성)
                    for col in sample_df.columns:
                        try:
                            sample_df[col] = sample_df[col].astype(str)
                        except:
                            sample_df[col] = "변환실패"

                    content += "샘플 데이터:\n"
                    content += str(sample_df)
                except Exception as e:
                    content += f"샘플 데이터 처리 실패: {e}"

                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "sheet": sheet_name,
                        "type": "excel",
                        "engine": successful_engine,
                        "rows": len(df),
                        "columns": len(df.columns)
                    }
                ))

            except Exception as e:
                print(f"⚠️ 시트 {sheet_name} 처리 실패: {e}")
                documents.append(Document(
                    page_content=f"Excel 시트 처리 실패: {file_path.name} - {sheet_name}\n오류: {str(e)}",
                    metadata={"source": str(file_path), "sheet": sheet_name, "type": "excel_error"}
                ))

        return documents

    def build_vectorstore(self, force_rebuild=False, use_alternative_path=False):
        """벡터스토어 구축"""
        if use_alternative_path:
            # PyCharm 파일 잠금 문제 해결을 위한 대안 경로
            import tempfile
            vectorstore_path = Path(tempfile.gettempdir()) / "chroma_db_v6_military"
            print(f"📁 대안 경로 사용: {vectorstore_path}")
        else:
            vectorstore_path = self.db_path / "chroma_db_v6_military"

        if vectorstore_path.exists() and not force_rebuild:
            print("📚 기존 벡터스토어를 사용합니다")
            # 선택한 경로의 DB를 실제로 열어 RAG 체인을 구성한다 (대안 경로는 시작 시 자동 로드되지 않았을 수 있음).
            self.vectorstore = Chroma(
                persist_directory=str(vectorstore_path),
                embedding_function=self.embeddings
            )
            self.setup_rag_chain()
            return True

        print("🔨 벡터스토어 구축 중...")

        # 문서 로드
        documents = self.load_documents_from_folder()

        if not documents:
            print("❌ 처리할 문서가 없습니다")
            return False

        # 텍스트 분할 - 군사 데이터 최적화
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", "===", "---", ". ", " ", ""]
        )

        splits = text_splitter.split_documents(documents)
        print(f"📄 {len(splits):,}개 청크로 분할 완료")

        try:
            # 기존 디렉토리 삭제
            if vectorstore_path.exists():
                import shutil
                shutil.rmtree(vectorstore_path)

            # 벡터스토어 생성
            self.vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=self.embeddings,
                persist_directory=str(vectorstore_path)
            )

            print("✅ 벡터스토어 구축 완료!")

        except Exception as e:
            print(f"❌ 벡터스토어 구축 실패: {e}")
            return False

        # RAG 체인 설정
        self.setup_rag_chain()
        return True

    def load_vectorstore(self):
        """기존 벡터스토어 로드 (기본 경로 우선, 없으면 대안 경로)"""
        import tempfile
        vectorstore_path = self.db_path / "chroma_db_v6_military"
        alternative_path = Path(tempfile.gettempdir()) / "chroma_db_v6_military"
        if not vectorstore_path.exists() and alternative_path.exists():
            print(f"📁 대안 경로의 벡터스토어를 사용합니다: {alternative_path}")
            vectorstore_path = alternative_path

        if vectorstore_path.exists():
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(vectorstore_path),
                    embedding_function=self.embeddings
                )
                self.setup_rag_chain()
                print("✅ 기존 벡터스토어 로드 완료")

            except Exception as e:
                print(f"⚠️ 벡터스토어 로드 실패: {e}")
                print("새로 구축합니다...")
                self.build_vectorstore(force_rebuild=True)

    def add_new_data_to_vectorstore(self, new_documents):
        """기존 벡터스토어에 새 데이터만 추가하는 증분 업데이트"""

        if not self.vectorstore:
            print("❌ 벡터스토어가 초기화되지 않았습니다. 먼저 전체 구축을 실행하세요.")
            return False

        if not new_documents:
            print("📭 추가할 새 문서가 없습니다.")
            return False

        try:
            print(f"📄 {len(new_documents)}개의 새 문서를 처리합니다...")

            # 기존 소스 파일 목록 가져오기
            existing_sources = set()
            try:
                # Chroma에서 기존 메타데이터 가져오기
                collection = self.vectorstore._collection
                results = collection.get(include=['metadatas'])

                for metadata in results['metadatas']:
                    if metadata and 'source' in metadata:
                        existing_sources.add(metadata['source'])

                print(f"📚 기존 소스: {len(existing_sources)}개")

            except Exception as e:
                print(f"⚠️ 기존 소스 확인 실패: {e}")
                # 실패해도 계속 진행

            # 새 문서만 필터링
            truly_new_documents = []
            duplicate_count = 0

            for doc in new_documents:
                source = doc.metadata.get('source', '')
                if source and source not in existing_sources:
                    truly_new_documents.append(doc)
                else:
                    duplicate_count += 1

            if not truly_new_documents:
                print(f"📭 모든 문서가 이미 존재합니다. (중복: {duplicate_count}개)")
                return True  # 성공으로 처리

            print(f"🆕 {len(truly_new_documents)}개의 새로운 문서 발견 (중복 제외: {duplicate_count}개)")

            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", "===", "---", ". ", " ", ""]
            )

            new_splits = text_splitter.split_documents(truly_new_documents)
            print(f"✂️ {len(new_splits)}개 청크로 분할 완료")

            # 벡터스토어에 추가
            print("🔄 벡터스토어에 추가 중...")

            # 배치 크기 설정 (메모리 관리)
            batch_size = 100
            total_added = 0

            for i in range(0, len(new_splits), batch_size):
                batch = new_splits[i:i + batch_size]

                try:
                    # Chroma의 add_documents 메서드 사용
                    self.vectorstore.add_documents(batch)
                    total_added += len(batch)
                    print(f"   ✅ {total_added}/{len(new_splits)} 청크 추가됨...")

                except Exception as e:
                    print(f"⚠️ 배치 추가 실패: {e}")
                    # 개별 추가 시도
                    for doc in batch:
                        try:
                            self.vectorstore.add_documents([doc])
                            total_added += 1
                        except Exception as e2:
                            print(f"❌ 개별 문서 추가 실패: {e2}")

            print(f"✅ 증분 업데이트 완료! 총 {total_added}개 청크 추가됨")

            # 새로운 소스 목록
            new_sources = set()
            for doc in truly_new_documents:
                source = doc.metadata.get('source', '')
                if source:
                    new_sources.add(Path(source).name)

            if new_sources:
                print(f"📁 추가된 파일: {', '.join(list(new_sources)[:5])}")
                if len(new_sources) > 5:
                    print(f"   ... 외 {len(new_sources) - 5}개")

            # RAG 체인은 그대로 유지 (retriever가 자동으로 업데이트된 데이터 사용)

            return True

        except Exception as e:
            print(f"❌ 증분 업데이트 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False

    def get_existing_files_from_vectorstore(self):
        """벡터스토어에서 이미 처리된 파일 목록 가져오기"""

        if not self.vectorstore:
            return set()

        existing_files = set()

        try:
            collection = self.vectorstore._collection

            # 모든 메타데이터 가져오기 (청크 수가 많을 수 있으므로 제한)
            # Chroma의 get() 메서드는 기본적으로 모든 데이터를 반환
            results = collection.get(
                include=['metadatas'],
                limit=10000  # 충분히 큰 수로 설정
            )

            for metadata in results['metadatas']:
                if metadata and 'source' in metadata:
                    source_path = metadata['source']
                    # 파일명만 저장 (전체 경로가 달라질 수 있으므로)
                    filename = Path(source_path).name
                    existing_files.add(filename)

            print(f"📚 벡터스토어에서 {len(existing_files)}개의 고유 파일 발견")

        except Exception as e:
            print(f"⚠️ 기존 파일 목록 가져오기 실패: {e}")

        return existing_files

    def load_only_new_documents(self):
        """새로운 문서만 로드하는 메서드"""

        # 기존에 처리된 파일 목록 가져오기
        existing_files = self.get_existing_files_from_vectorstore()

        # 전체 문서 로드
        all_documents = self.load_documents_from_folder()

        # 새 문서만 필터링
        new_documents = []
        skipped_count = 0

        for doc in all_documents:
            source = doc.metadata.get('source', '')
            if source:
                filename = Path(source).name
                if filename not in existing_files:
                    new_documents.append(doc)
                else:
                    skipped_count += 1

        print(f"🆕 새 문서: {len(new_documents)}개")
        print(f"⏭️ 건너뛴 문서: {skipped_count}개 (이미 처리됨)")

        return new_documents

    def setup_rag_chain(self):
        """RAG 체인 설정 - 군사 전문 분석"""
        if self.vectorstore:
            # retriever 설정
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 8}  # 더 많은 문서 검색
            )

            # 문서 포맷 함수
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            self.format_docs = format_docs
            # 이미 검색한 문서로 답변만 생성하는 체인 (ask()에서 검색을 한 번만 하기 위해 사용)
            self.answer_chain = self.current_template | self.llm | StrOutputParser()

            # 군사 전문 RAG 체인
            self.rag_chain = (
                    {
                        "context": self.retriever | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | self.current_template
                    | self.llm
                    | StrOutputParser()
            )

            print("✅ 군사·전력 분석 RAG 체인 구성 완료")

    def configure_personal_scraping(self, max_pages_total=1000, max_pages_per_domain=100,
                                    max_depth=3, delay=1, content_types="all"):
        """개인 스크래핑 설정"""
        self.personal_scraper.configure_scraping_limits(
            max_pages_total=max_pages_total,
            max_pages_per_domain=max_pages_per_domain,
            max_depth=max_depth,
            delay_between_requests=delay
        )

        self.scraping_content_types = content_types
        print(f"✅ 스크래핑 설정 완료: 총 {max_pages_total}페이지, 도메인당 {max_pages_per_domain}페이지")

    def scrape_personal_data(self, start_urls):
        """개인 자료 수집용 스크래핑"""
        if not start_urls:
            return {'success': 0, 'total': 0, 'message': 'URL이 없습니다'}

        print(f"🔍 개인 자료 수집 시작: {len(start_urls)}개 URL")

        # scraping_content_types 속성 확인 및 초기화
        if not hasattr(self, 'scraping_content_types'):
            self.scraping_content_types = "all"

        # 스크래핑 실행
        scrape_results = self.personal_scraper.scrape_any_content_type(
            start_urls,
            self.scraping_content_types
        )

        # 성공한 결과만 문서로 변환
        web_documents = []
        success_count = 0

        for result in scrape_results:
            if result.get('success') and result.get('content', '').strip():
                web_content = f"웹페이지 ({result.get('content_type', 'unknown')}): {result.get('title', 'Unknown Title')}\n"
                web_content += f"URL: {result['url']}\n"
                web_content += f"콘텐츠 타입: {result.get('content_type', 'unknown')}\n"
                web_content += f"수집 방법: {result.get('method', 'unknown')}\n"
                web_content += f"수집 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

                # 메타데이터 추가
                if result.get('metadata'):
                    for key, value in result['metadata'].items():
                        if value:
                            web_content += f"{key}: {value}\n"
                    web_content += "\n"

                web_content += result['content']

                web_documents.append(Document(
                    page_content=web_content,
                    metadata={
                        'source': result['url'],
                        'type': 'web_page_personal',
                        'content_type': result.get('content_type', 'unknown'),
                        'method': result.get('method', 'unknown'),
                        'scraped_time': datetime.now().isoformat()
                    }
                ))
                success_count += 1

        if web_documents and self.vectorstore:
            # 텍스트 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100,
                separators=["\n\n", "\n", ". ", " ", ""]
            )

            splits = text_splitter.split_documents(web_documents)

            # 벡터스토어에 추가
            self.vectorstore.add_documents(splits)
            print(f"📄 {len(splits)}개 청크 추가됨")

        return {
            'success': success_count,
            'total': len(start_urls),
            'new_documents': len(web_documents),
            'unique_domains': len(self.personal_scraper.domain_stats),
            'message': f'{success_count}/{len(start_urls)}개 URL 스크래핑 성공, {len(web_documents)}개 문서 추가'
        }

    def ask(self, question):
        """질문에 답변 - 군사·전력 전문 분석"""
        if not self.rag_chain:
            return "❌ RAG 시스템이 초기화되지 않았습니다. 먼저 벡터스토어를 구축해주세요."

        try:
            # 질문 전처리 - 한국어 국가명 영어로 변환
            processed_question = self.preprocess_question(question)

            # 검색된 문서 확인
            retrieved_docs = self.retriever.invoke(processed_question)

            # 위에서 검색한 문서로 답변 생성 (rag_chain.invoke는 같은 검색을 한 번 더 수행하므로 쓰지 않음)
            response = self.answer_chain.invoke({
                "context": self.format_docs(retrieved_docs),
                "question": processed_question,
            })

            # 출처 정보 추가
            sources = []
            web_sources = []
            economic_sources = []

            for doc in retrieved_docs:
                source = doc.metadata.get('source', 'Unknown')
                doc_type = doc.metadata.get('type', 'file')

                if doc_type == 'web_page_personal':
                    content_type = doc.metadata.get('content_type', 'unknown')
                    web_sources.append(f"{content_type}: {source}")
                elif doc_type == 'economic_data':
                    country = doc.metadata.get('country', '')
                    data_type = doc.metadata.get('data_type', '')
                    economic_sources.append(f"{country} {data_type}")
                else:
                    sources.append(Path(source).name)

            source_info = "\n\n📚 참고 자료:"
            if economic_sources:
                source_info += f"\n💰 경제 데이터: {', '.join(set(economic_sources))}"
            if sources:
                source_info += f"\n📁 파일: {', '.join(set(sources))}"
            if web_sources:
                source_info += f"\n🌐 웹페이지: {len(set(web_sources))}개 ({', '.join(list(set(web_sources))[:3])}...)"

            return response + source_info

        except Exception as e:
            return f"❌ 오류 발생: {e}"

    def preprocess_question(self, question):
        """질문 전처리 - 한영 매핑"""
        processed = question

        # 한국어 국가명을 영어로 변환
        for korean, english in self.economic_processor.country_mapping.items():
            if korean in processed:
                processed += f" {english}"

        # 군사 관련 키워드 추가
        military_keywords = {
            'OOB': 'order of battle force structure 편제',
            '전력': 'military force combat power',
            '무기': 'weapons armament',
            '장비': 'equipment military hardware',
            '편제': 'order of battle organization',
            '부대': 'unit formation military',
            '병력': 'manpower troops personnel',
            '작전': 'operation military campaign',
            '전투': 'battle combat engagement'
        }

        for korean, english_terms in military_keywords.items():
            if korean in question:
                processed += f" {english_terms}"

        return processed

    def get_stats(self):
        """시스템 통계 - 안전한 속성 접근"""
        try:
            stats = f"📊 군사·전력 분석 전문 RAG 시스템 v6.0 통계\n\n"

            # 벡터스토어 상태 확인
            if not self.vectorstore:
                stats += "❌ 벡터스토어: 구축되지 않음\n"
                stats += "📋 시스템을 초기화하고 지식베이스를 구축해주세요.\n\n"
            else:
                try:
                    collection = self.vectorstore._collection
                    count = collection.count()
                    stats += f"📄 총 청크 수: {count:,}개\n"
                except Exception as e:
                    stats += f"📄 총 청크 수: 확인 불가 ({str(e)[:30]}...)\n"

            # LLM 모델 정보
            if hasattr(self, 'llm') and self.llm:
                try:
                    model_name = getattr(self.llm, 'model', 'Unknown')
                    stats += f"🤖 LLM 모델: {model_name} (로컬)\n"
                except:
                    stats += f"🤖 LLM 모델: 연결됨 (로컬)\n"
            else:
                stats += f"🤖 LLM 모델: 연결되지 않음\n"

            # 임베딩 모델 정보
            if hasattr(self, 'embeddings') and self.embeddings:
                try:
                    embedding_name = getattr(self.embeddings, 'model_name', 'Unknown')
                    stats += f"🔍 임베딩 모델: {embedding_name} (로컬)\n"
                except:
                    stats += f"🔍 임베딩 모델: 로드됨 (로컬)\n"
            else:
                stats += f"🔍 임베딩 모델: 로드되지 않음\n"

            stats += f"🎯 전문 분야: 군사·전력·국가모의전 분석\n"
            stats += f"🌐 웹 스크래핑: 개인 자료 수집 특화\n"
            stats += f"💰 경제 데이터: GDP 전용 처리기 탑재\n"
            stats += f"🔄 중복 제거: 지능형 중복 감지\n"

            # 경로 정보
            if hasattr(self, 'data_path'):
                stats += f"📁 데이터 폴더: {self.data_path}\n"
            if hasattr(self, 'db_path'):
                stats += f"💾 벡터스토어: Chroma v6.0 (로컬)\n\n"

            stats += f"🎯 현재 템플릿: 군사·전력 분석 전문\n"
            stats += f"🔄 분석 모드: OOB/무기체계/작전분석 특화\n\n"

            # 스크래핑 통계
            if (hasattr(self, 'personal_scraper') and
                    hasattr(self.personal_scraper, 'domain_stats') and
                    self.personal_scraper.domain_stats):
                stats += f"🕷️ 스크래핑 통계:\n"
                for domain, count in list(self.personal_scraper.domain_stats.items())[:5]:
                    stats += f"   - {domain}: {count}페이지\n"

                total_scraped = sum(self.personal_scraper.domain_stats.values())
                stats += f"   총 수집: {total_scraped}페이지\n\n"

            # 중복 제거 통계
            if (hasattr(self, 'personal_scraper') and
                    hasattr(self.personal_scraper, 'duplicate_detector')):
                try:
                    unique_count = len(self.personal_scraper.duplicate_detector.content_hashes)
                    stats += f"🔍 중복 제거: {unique_count}개 고유 콘텐츠\n\n"
                except:
                    stats += f"🔍 중복 제거: 활성화됨\n\n"

            # GPU 정보
            try:
                import torch
                if torch.cuda.is_available():
                    current_device = torch.cuda.current_device()
                    gpu_name = torch.cuda.get_device_name(current_device)
                    stats += f"🎮 GPU: {gpu_name}\n"
                else:
                    stats += f"🎮 GPU: 미사용 (CPU 모드)\n"
            except Exception:
                stats += f"🎮 GPU: 상태 확인 불가\n"

            return stats

        except Exception as e:
            return f"📊 시스템 통계\n\n❌ 통계 수집 오류: {str(e)}\n\n기본 상태:\n- 시스템 초기화 필요\n- 벡터스토어 구축 필요"


def create_v6_military_interface():
    """v6.0 군사·전력 분석 전문 Gradio 인터페이스"""

    rag_system = None

    def initialize_v6_system(llm_model):
        nonlocal rag_system
        try:
            rag_system = UltimateMilitaryRAGv6(llm_model=llm_model)
            return "✅ v6.0 군사·전력 분석 시스템 초기화 완료!", rag_system.get_stats()
        except Exception as e:
            return f"❌ 초기화 실패: {e}", ""

    def build_knowledge_base(force_rebuild, use_alternative_path):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요.", ""

        try:
            success = rag_system.build_vectorstore(
                force_rebuild=force_rebuild,
                use_alternative_path=use_alternative_path
            )
            if success:
                return "✅ 지식 베이스 구축 완료!", rag_system.get_stats()
            else:
                return "❌ 지식 베이스 구축 실패 (로그 확인)", rag_system.get_stats()
        except Exception as e:
            return f"❌ 구축 실패: {e}", ""

    def add_new_files_only():
        """새 파일만 기존 벡터스토어에 추가"""
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요.", ""

        try:
            # 새 문서들만 로드 (이미 있는 파일은 제외)
            new_documents = rag_system.load_only_new_documents()

            if not new_documents:
                return "📭 추가할 새 문서가 없습니다", rag_system.get_stats()

            # 기존 벡터스토어에 추가
            success = rag_system.add_new_data_to_vectorstore(new_documents)

            if success:
                return f"✅ {len(new_documents)}개 새 문서 증분 추가 완료!", rag_system.get_stats()
            else:
                return "❌ 증분 추가 실패, 전체 재구축 필요", rag_system.get_stats()

        except Exception as e:
            return f"❌ 증분 추가 실패: {e}", rag_system.get_stats()

    def configure_scraping(max_total, max_domain, max_depth, delay, content_types):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요."

        try:
            rag_system.configure_personal_scraping(
                max_pages_total=max_total,
                max_pages_per_domain=max_domain,
                max_depth=max_depth,
                delay=delay,
                content_types=content_types
            )
            return f"✅ 스크래핑 설정 완료: 총 {max_total}페이지, 도메인당 {max_domain}페이지, 깊이 {max_depth}"
        except Exception as e:
            return f"❌ 설정 실패: {e}"

    def scrape_personal_content(urls_text):
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화해주세요.", ""

        if not urls_text.strip():
            return "URL을 입력해주세요.", ""

        try:
            urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
            result = rag_system.scrape_personal_data(urls)
            return result['message'], rag_system.get_stats()
        except Exception as e:
            return f"❌ 스크래핑 실패: {e}", rag_system.get_stats() if rag_system else ""

    def answer_military_question(question):
        """군사·전력 분석 질문 답변"""
        nonlocal rag_system
        if not rag_system:
            return "❌ 먼저 시스템을 초기화하고 지식 베이스를 구축해주세요."

        if not question.strip():
            return "질문을 입력해주세요."

        return rag_system.ask(question)

    # Gradio 인터페이스
    with gr.Blocks(title="군사·전력 분석 전문 RAG 시스템 v6.0", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🎯 군사·전력 분석 전문 RAG 시스템 v6.0

        ### 🎖️ v6.0 전문 분야:
        - 🏛️ **OOB(전력 편성)**: 특정 연도·국가/군관구/부대의 병력·장비 수량, 지휘체계, 배치
        - ⚔️ **무기 체계/사양**: 포/전차/장갑차/박격포/기관총/항공/전자전 등 모델·구경·성능 비교
        - 🏭 **생산·조달 가능성**: 단기간 대량 도입의 현실성 평가, 산업동원 능력
        - 📊 **연표·작전 서술**: 스페인 내전, 터키 작전, 우크라이나 전역 등 내러티브 히스토리
        - 💰 **손실/대가 평가**: 특정 지역 확보 대가로의 군사·정치·경제 손실
        - 🌍 **대안사 분기**: 정책·결정 변화에 따른 이벤트 카드식 시나리오

        ### 📋 답변 원칙:
        - ✅ **정량 우선**: 병력·장비 수량, 구경, 사거리, 생산량/속도 등 숫자 제시
        - 📅 **절대 날짜**: YYYY-MM-DD 형식으로 정확한 날짜 표기
        - 🔍 **현실성 검증**: 산업·훈련·정비·보급 제약을 근거로 실행 가능성 평가
        - 🕊️ **인도주의 준수**: 민간 피해 최소화 전제, ROE/IHL 고려
        - 📖 **출처 명시**: 신뢰 가능한 1차 사료 우선, 불확실성 범위 표기
        """)

        with gr.Tab("⚙️ 시스템 설정"):
            gr.Markdown("""
            ### 🔧 v6.0 군사·전력 분석 시스템 초기화
            - 🎯 군사 전문 프롬프트 탑재
            - 📊 OOB/무기체계/작전분석 특화 
            - 🔍 향상된 군사 용어 검색
            - 💬 전문가 수준 분석 제공

            ### 🛠️ PyCharm 파일 잠금 문제 해결책:
            - **증분 업데이트**: 새 데이터만 추가 (재구축 없음)
            - **대안 경로 사용**: 임시 폴더에 DB 생성
            - **안전한 제거**: 파일 핸들 해제 후 제거

            ### 📋 사용 가능한 모델:
            - **qwen2.5:7b**: 균형 잡힌 성능, 빠른 응답 (권장)
            - **gemma3n:e2b**: 경량화된 모델, 가장 빠른 응답
            """)

            llm_model_choice = gr.Dropdown(
                label="LLM 모델 선택",
                choices=["qwen2.5:7b", "gemma3n:e2b"],  # 사용자가 가진 모델만
                value="qwen2.5:7b"  # 기본값을 qwen2.5:7b로 설정
            )

            with gr.Row():
                init_btn = gr.Button("🚀 v6.0 시스템 초기화", variant="primary")

            with gr.Row():
                build_btn = gr.Button("🔨 완전 재구축", variant="secondary")
                incremental_btn = gr.Button("📄 증분 추가 (새 파일만)", variant="secondary")

            with gr.Row():
                alt_path_checkbox = gr.Checkbox(
                    label="대안 경로 사용 (PyCharm 문제 해결)",
                    value=False
                )

            init_output = gr.Textbox(label="초기화 상태", lines=2)
            system_stats = gr.Textbox(label="시스템 정보", lines=20)

            init_btn.click(
                initialize_v6_system,
                inputs=llm_model_choice,
                outputs=[init_output, system_stats]
            )

            build_btn.click(
                lambda use_alt: build_knowledge_base(True, use_alt),
                inputs=alt_path_checkbox,
                outputs=[init_output, system_stats]
            )

            incremental_btn.click(
                add_new_files_only,
                outputs=[init_output, system_stats]
            )

        with gr.Tab("🎯 군사·전력 분석"):
            gr.Markdown("""
            ### 🎖️ 군사·전력 전문 분석

            **이런 질문들에 전문적으로 답변합니다:**
            - 🏛️ **OOB 질문**: "2016년 러시아 서부군관구 편제 정리해줘"
            - ⚔️ **무기 비교**: "T-72B3와 T-80BVM 사양 비교분석"  
            - 🏭 **생산 가능성**: "1936년 캐나다가 6개월에 포 300문 생산 가능?"
            - 📊 **작전 분석**: "과달라하라 전투를 내러티브로 정리"
            - 💰 **손실 평가**: "드니프로 동쪽 확보시 러시아·우크라이나 양측 대가는?"
            - 🌍 **대안사**: "황도파 집권 시나리오의 군사적 파급효과"

            **답변 형식:**
            - 📋 요약 → 핵심 표(OOB/제원/생산) → 상세 분석 → 출처

            **정량 중심:** 모든 수치와 날짜를 정확하게 제시
            """)

            military_question_input = gr.Textbox(
                label="군사·전력 분석 질문",
                placeholder="예: 1936년 스페인 내전 당시 공화파 전력 편성은? 또는 2022년 우크라이나 전쟁 초기 러시아군 BTG 구성은?",
                lines=4
            )

            military_ask_btn = gr.Button("🎯 전문 분석 받기", variant="primary", size="lg")

            military_answer_output = gr.Textbox(
                label="군사·전력 전문 분석 결과",
                lines=30
            )

            military_ask_btn.click(
                answer_military_question,
                inputs=military_question_input,
                outputs=military_answer_output
            )

            gr.Examples(
                examples=[
                    "2016년 러시아 서부군관구 편제와 주요 부대 현황 정리해줘",
                    "T-72B3와 T-80BVM의 사양을 표로 비교 분석해줘",
                    "1936년 캐나다가 6개월에 야포 300문을 생산할 수 있었을까?",
                    "과달라하라 전투를 시간순으로 내러티브 형태로 정리해줘",
                    "드니프로 강 동쪽 확보를 위한 러시아군의 예상 손실은?",
                    "1936년 파라과이군의 중장비 전력 현황은?",
                    "바그너 그룹의 2016년 조직도와 지휘체계는?",
                    "황도파 집권 시나리오에서 일본군 확장 가능성은?",
                    "스페인 내전 당시 독일 콘도르 군단의 장비와 전력은?",
                    "1991년 유고슬라비아 인민군(JNA) 편제 현황은?"
                ],
                inputs=military_question_input
            )

        with gr.Tab("🌐 개인 자료 수집"):
            gr.Markdown("""
            ### 📚 군사·전력 관련 웹페이지 수집

            **🚫 자동으로 제외되는 쓸모없는 링크:**
            - 💰 **광고/후원**: PayPal, 도네이션, 스폰서 링크
            - 🛒 **쇼핑몰**: 구매, 장바구니, 상품 페이지
            - 🔐 **로그인/가입**: 로그인, 회원가입 필수 페이지
            - ❌ **에러/만료**: 404, 도메인 만료, 에러 페이지
            - 🎰 **스팸/도박**: 카지노, 베팅, 의심스러운 콘텐츠

            **✅ 수집되는 유용한 군사 콘텐츠:**
            - 📖 **학술자료**: 군사사 논문, 연구보고서
            - 📰 **뉴스**: 국방 관련 기사, 분석글
            - 💬 **포럼**: 군사 커뮤니티, 전문가 토론
            - 🎓 **위키**: 무기체계, 전투사 자료
            - 📊 **공식문서**: 정부/군 발표자료
            """)

            with gr.Row():
                with gr.Column():
                    max_total_pages = gr.Slider(
                        label="총 최대 페이지 수",
                        minimum=10,
                        maximum=10000,
                        value=1000,
                        step=10
                    )

                    max_domain_pages = gr.Slider(
                        label="도메인당 최대 페이지 수",
                        minimum=5,
                        maximum=1000,
                        value=100,
                        step=5
                    )

                with gr.Column():
                    max_crawl_depth = gr.Slider(
                        label="최대 탐색 깊이 (링크 따라가기)",
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1
                    )

                    request_delay = gr.Slider(
                        label="요청 간 지연 (초)",
                        minimum=0.5,
                        maximum=5.0,
                        value=1.0,
                        step=0.5
                    )

            content_type_choice = gr.Dropdown(
                label="수집할 콘텐츠 타입",
                choices=["all", "academic", "news", "forum", "reference"],
                value="all"
            )

            config_btn = gr.Button("⚙️ 스크래핑 설정 적용", variant="secondary")
            config_output = gr.Textbox(label="설정 상태", lines=2)

            urls_input = gr.Textbox(
                label="수집할 군사·전력 관련 URL 목록 (한 줄에 하나씩)",
                placeholder="""https://defense-blog.example.com
https://military-history.example.com
https://weapons-forum.example.com
https://research.defense.gov""",
                lines=8
            )

            scrape_btn = gr.Button("🔍 군사 자료 수집 시작", variant="primary")
            scrape_output = gr.Textbox(label="수집 결과 (필터링 통계 포함)", lines=5)
            scrape_stats = gr.Textbox(label="업데이트된 통계", lines=15)

            config_btn.click(
                configure_scraping,
                inputs=[max_total_pages, max_domain_pages, max_crawl_depth, request_delay, content_type_choice],
                outputs=config_output
            )

            scrape_btn.click(
                scrape_personal_content,
                inputs=urls_input,
                outputs=[scrape_output, scrape_stats]
            )

        with gr.Tab("📊 시스템 현황"):
            stats_btn = gr.Button("📊 최신 통계 확인", variant="secondary")
            stats_output = gr.Textbox(label="시스템 통계", lines=25)

            stats_btn.click(
                lambda: rag_system.get_stats() if rag_system else "시스템이 초기화되지 않았습니다.",
                outputs=stats_output
            )

    return interface


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🎯 군사·전력 분석 전문 RAG 시스템 v6.0")
    print("=" * 80)

    # LangChain 패키지 상태 확인
    print("📦 LangChain 패키지 상태:")
    if OLLAMA_NEW:
        print("✅ langchain-ollama: 최신 버전 사용")
    else:
        print("⚠️ langchain-ollama: 구버전 (업그레이드 권장)")

    print("\n🎖️ v6.0 군사·전력 분석 전문 기능:")
    print("- 🏛️ OOB(전력 편성): 특정 연도·국가/군관구/부대의 병력·장비 수량, 지휘체계, 배치")
    print("- ⚔️ 무기 체계/사양: 포/전차/장갑차/박격포/기관총/항공/전자전 등 모델·구경·성능 비교")
    print("- 🏭 생산·조달 가능성: 단기간 대량 도입의 현실성 평가, 산업동원 능력")
    print("- 📊 연표·작전 서술: 스페인 내전, 터키 작전, 우크라이나 전역 등 내러티브 히스토리")
    print("- 💰 손실/대가 평가: 특정 지역 확보 대가로의 군사·정치·경제 손실")
    print("- 🌍 대안사 분기: 정책·결정 변화에 따른 이벤트 카드식 시나리오")

    print("\n📋 답변 원칙:")
    print("- ✅ 정량 우선: 병력·장비 수량, 구경, 사거리, 생산량/속도 등 숫자 제시")
    print("- 📅 절대 날짜: YYYY-MM-DD 형식으로 정확한 날짜 표기")
    print("- 🔍 현실성 검증: 산업·훈련·정비·보급 제약을 근거로 실행 가능성 평가")
    print("- 🕊️ 인도주의 준수: 민간 피해 최소화 전제, ROE/IHL 고려")
    print("- 📖 출처 명시: 신뢰 가능한 1차 사료 우선, 불확실성 범위 표기")

    print("\n🎯 질문 예시:")
    print("- '2016년 러시아 서부군관구 편제와 주요 부대 현황 정리해줘'")
    print("- 'T-72B3와 T-80BVM의 사양을 표로 비교 분석해줘'")
    print("- '1936년 캐나다가 6개월에 야포 300문을 생산할 수 있었을까?'")
    print("- '과달라하라 전투를 시간순으로 내러티브 형태로 정리해줘'")
    print("- '드니프로 강 동쪽 확보를 위한 러시아군의 예상 손실은?'")
    print("- '바그너 그룹의 2016년 조직도와 지휘체계는?'")

    print("\n🚫 자동 필터링되는 쓸모없는 콘텐츠:")
    print("- 💰 광고/후원/쇼핑 링크 (PayPal, 도네이션 등)")
    print("- 🔐 로그인/가입 필수 페이지")
    print("- ❌ 404/만료/에러 페이지")
    print("- 🎰 스팸/도박/의심스러운 콘텐츠")
    print("- 🔧 기술적 파일 (CSS, JS, API 등)")

    print("\n📋 설치 명령어:")
    print("1. ollama pull qwen2.5:7b")
    print("2. pip install --upgrade sentence-transformers chromadb torch")
    print('3. pip install --upgrade langchain langchain-community "langchain-huggingface>=0.3.1,<2"')
    print("4. pip install pandas numpy openpyxl xlrd beautifulsoup4")
    print("5. pip install playwright trafilatura newspaper3k")

    print("\n🔄 최신 LangChain 패키지 (권장):")
    print("6. pip install -U langchain-ollama")
    print("7. pip install -U langchain-chroma")

    print("\n🤖 사용 가능한 모델:")
    print("- qwen2.5:7b: 균형 잡힌 성능, 빠른 응답 (권장)")
    print("- gemma3n:e2b: 경량화된 모델, 가장 빠른 응답")

    # 웹 인터페이스 실행
    interface = create_v6_military_interface()

    print(f"\n🚀 v6.0 군사·전력 분석 시스템 시작!")
    print("🌐 브라우저: http://localhost:7860")
    print("\n📁 사용법:")
    print("1. v6.0 시스템 초기화 (qwen2.5:7b 또는 gemma3n:e2b 선택)")
    print("2. '군사·전력 분석' 탭에서 전문 분석 요청!")
    print("   - OOB/무기체계/작전분석 등 전문 질문")
    print("   - 정량 데이터와 출처 명시된 답변 제공")
    print("3. 필요시 개인 자료 수집으로 군사 데이터 확장")

    print("\n💡 핵심 포인트:")
    print("- 🎯 군사·전력 분석 전문! OOB/무기/작전 특화")
    print("- 📊 정량 중심 분석 - 모든 수치와 날짜 정확하게")
    print("- 🔍 현실성 검증 - 산업·보급·훈련 제약 고려")
    print("- 📖 출처 명시 - 신뢰 가능한 1차 사료 우선")
    print("- 🕊️ 인도주의 원칙 준수")

    print("\n🔧 PyCharm 사용자:")
    print("- 증분 업데이트로 파일 잠금 문제 해결")
    print("- 대안 경로 옵션으로 권한 문제 회피")

    print("\n📋 답변 형식:")
    print("- 요약 → 핵심 표(OOB/제원/생산) → 상세 분석 → 출처")
    print("- 사실과 가정 명확히 분리")
    print("- 불확실성 범위와 신뢰도 표기")

    print("\n⚡ 모델별 특징:")
    print("- qwen2.5:7b: 더 정확하고 상세한 분석 (권장)")
    print("- gemma3n:e2b: 빠른 응답, 기본적인 분석")

    interface.launch(
        server_port=7860,
        share=False,
        inbrowser=True
    )