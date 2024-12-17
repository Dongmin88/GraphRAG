GraphRAG 시스템
개요
이 프로젝트는 PDF 문서에서 지식을 추출하고 그래프 기반의 검색 증강 생성(Graph-based Retrieval-Augmented Generation, GraphRAG)을 수행하는 시스템입니다. Llama 3 모델을 기반으로 하여 문서의 내용을 이해하고 질문에 답변할 수 있습니다.
주요 기능

PDF 문서 처리 및 텍스트 추출
지식 그래프 구성
의미 기반 검색
질의응답 생성

설치 방법
필수 요구사항

Python 3.7 이상
CUDA 지원 GPU (권장)

필요한 패키지 설치
bashCopypip install torch transformers networkx numpy huggingface_hub pdfplumber tqdm spacy tkinter
python -m spacy download en_core_web_sm
Hugging Face 토큰 설정
pythonCopytoken = ""  # 여기에 본인의 토큰을 입력하세요
파일 구조
Copyproject_folder/
    ├── graphrag.py        # GraphRAG 핵심 클래스 및 기능
    ├── graphrag_gui.py    # 그래픽 사용자 인터페이스
    └── README.md         # 문서
사용 방법
1. GUI 모드 실행
bashCopypython graphrag_gui.py
2. 코드에서 직접 사용
pythonCopyfrom graphrag import PDFLlama3GraphRAG

# GraphRAG 시스템 초기화
rag = PDFLlama3GraphRAG()

# PDF 파일 처리
documents = rag.read_pdf("example.pdf")

# 그래프 구성
rag.construct_graph(documents)

# 질의응답
query = "원하는 질문을 입력하세요"
relevant_nodes = rag.retrieve(query)
nodes = [node for node, _ in relevant_nodes]
subgraph = rag.get_subgraph(nodes)
response = rag.generate_response(query, subgraph)
print(response)
주요 클래스 및 메소드
PDFLlama3GraphRAG

__init__(): 시스템 초기화
read_pdf(): PDF 파일 읽기 및 텍스트 추출
construct_graph(): 지식 그래프 구성
retrieve(): 관련 노드 검색
generate_response(): 응답 생성

PDFGraphRAGApp

GUI 인터페이스 제공
PDF 파일 선택
질의응답 인터페이스
진행 상황 표시

성능 최적화

배치 처리를 통한 엔티티 추출 최적화
텍스트 전처리 개선
GPU 가속 활용

주의사항

PDF 파일의 크기와 수에 따라 처리 시간이 달라질 수 있습니다
대용량 PDF 처리 시 충분한 메모리가 필요합니다
긴 문서의 경우 청크 단위로 분할하여 처리됩니다

문제 해결
일반적인 문제

PDF 텍스트 추출 문제

해결: 텍스트 전처리 기능 사용

pythonCopyfrom graphrag import preprocess_text
processed_text = preprocess_text(raw_text)

메모리 부족 오류

해결: batch_size 조정

pythonCopyrag = PDFLlama3GraphRAG()
rag.batch_size = 16  # 기본값보다 작게 설정

GPU 메모리 부족

해결: 모델 설정 변경

pythonCopyrag.model = rag.model.to('cpu')  # CPU 모드로 전환


향후 개선 계획

다국어 지원 확대
모델 최적화
병렬 처리 기능 강화
메모리 사용량 최적화

라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
문의사항
문제가 발생하거나 추가 기능이 필요한 경우 Issues에 등록해 주세요.
