# PDF-GraphRAG: PDF 문서를 위한 그래프 기반 검색 증강 생성 시스템

PDF-GraphRAG는 PDF 문서를 처리하고 Llama 3 언어 모델을 사용하여 질의에 답변하는 그래프 기반 검색 증강 생성 시스템의 파이썬 구현체입니다. 이 구현은 ["GraphRAG: Unlocking LLM Knowledge for Zero-Shot Graph Question Answering"](https://arxiv.org/pdf/2404.16130) 연구를 기반으로 합니다.

## 주요 기능

- PDF 문서 처리 및 청크 분할
- 문서 내용 기반 지식 그래프 구성
- Llama 3를 활용한 엔티티 추출
- 그래프 기반 정보 검색
- 사용자 친화적 GUI 인터페이스
- 멀티스레드 처리 지원
- 출처 추적 및 인용

## 시스템 요구사항

- Python 3.8 이상
- PyTorch
- Hugging Face Transformers
- NetworkX
- PDFPlumber
- Tkinter (GUI용)
- CUDA 지원 GPU (권장)

## 프로젝트 구조

- `graphrag.py`: GraphRAG 시스템 핵심 구현체
- `graphrag_gui.py`: 시스템 GUI 인터페이스
- `requirements.txt`: Python 의존성 목록

## 사용 방법

### 명령줄 인터페이스

```python
from graphrag import PDFLlama3GraphRAG

# 시스템 초기화
rag = PDFLlama3GraphRAG()

# PDF 문서 처리
documents = rag.read_pdf("문서경로/문서.pdf")
rag.construct_graph(documents)

# 시스템에 질의
query = "질문을 입력하세요"
relevant_nodes = rag.retrieve(query)
subgraph = rag.get_subgraph([node for node, _ in relevant_nodes])
response = rag.generate_response(query, subgraph)
print(response)
```

### GUI 인터페이스

```bash
python graphrag_gui.py
```

GUI는 다음과 같은 직관적인 인터페이스를 제공합니다:
1. PDF 파일 선택
2. 문서 처리
3. 질의 제출
4. 출처가 포함된 응답 확인

## 작동 방식

1. **PDF 처리**: 문서를 문맥과 메타데이터를 유지하면서 관리 가능한 크기로 분할합니다.

2. **엔티티 추출**: Llama 3 모델을 사용하여 텍스트 청크에서 주요 엔티티와 개념을 식별합니다.

3. **그래프 구성**: 다음과 같은 지식 그래프를 구축합니다:
   - 노드는 엔티티와 개념을 표현
   - 엣지는 함께 등장하는 엔티티 간의 관계를 표현
   - 노드 임베딩은 Llama 3를 사용하여 생성

4. **질의 처리**:
   - 질의를 동일한 모델을 사용하여 임베딩
   - 의미적 유사도를 사용하여 관련 노드 검색
   - 검색된 노드로부터 서브그래프 구성
   - 질의와 그래프 문맥을 고려하여 최종 응답 생성

5. **출처 추적**: 모든 응답에는 원본 PDF 문서와 페이지 번호에 대한 인용이 포함됩니다.

## 주요 컴포넌트

### PDFLlama3GraphRAG

GraphRAG 시스템을 구현하는 핵심 클래스:
- Llama 3 모델 초기화
- PDF 문서 처리
- 지식 그래프 관리
- 질의 처리

### PDFGraphRAGApp

다음 기능을 제공하는 GUI 애플리케이션:
- 파일 선택 인터페이스
- 진행 상황 추적
- 질의 입력
- 응답 표시
- 멀티스레드 처리

## 기여하기

기여는 언제나 환영합니다! Pull Request를 제출해 주세요.

## 인용

연구에서 이 구현체를 사용하실 경우 다음을 인용해 주세요:

```bibtex
@article{graphrag2024,
  title={GraphRAG: Unlocking LLM Knowledge for Zero-Shot Graph Question Answering},
  author={Original Authors},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024}
}
```

## 라이선스

MIT License

Copyright (c) 2024 [프로젝트 소유자]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## 한계점 및 향후 계획

- 현재는 PDF 문서만 지원
- Llama 3 모델의 컨텍스트 윈도우 제한
- 큰 문서의 경우 그래프 구성에 많은 메모리 필요
- 향후 개선 사항:
  - 더 많은 문서 형식 지원
  - 개선된 엔티티 추출
  - 더 정교한 그래프 구성 알고리즘
  - 대용량 문서 처리 개선