# LLM 바늘 찾기 테스트 (Needle Threading Test)

이 프로젝트는 대규모 언어 모델(LLM)의 키-값 검색 및 연쇄 추론 능력을 테스트하기 위한 프레임워크입니다. UUID 기반의 키-값 쌍을 사용하여 모델의 단일 검색과 연쇄 검색 능력을 평가합니다.

## 주요 기능

- UUID 기반의 무작위 키-값 쌍 생성
- 토큰 제한을 고려한 자동 데이터셋 크기 조절
- 단일 검색 테스트 (바늘 찾기)
- 연쇄 검색 테스트 (실타래 따라가기)
- 자동 성능 측정 및 결과 저장 기능

## 설치 방법

다음 명령어로 필요한 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```

### 필수 패키지
- transformers
- torch
- accelerate
- huggingface_hub

## 사용 방법

### 기본 실행 방법

```python
from needle_threading import NeedleThreadingTest

# 테스터 초기화
tester = NeedleThreadingTest(model_name="meta-llama/Llama-3.2-1B")

# 테스트 실행
results = tester.run_tests(num_trials=3)
```

### 출력 예시
```plaintext
Testing with 5 UUID pairs
===========================================

------------------------------
Trial 1/3
------------------------------

Single Needle Test:
Key: 123e4567-e89b-12d3-a456-426614174000
Expected: 987fcdeb-51a2-43f7-b145-87cf92a56321
Got: 987fcdeb-51a2-43f7-b145-87cf92a56321
Correct: True

Threading Test:
Thread start: 123e4567-e89b-12d3-a456-426614174000
Expected final: 456e7890-c12b-34d5-e678-912345678901
Got: 456e7890-c12b-34d5-e678-912345678901
Correct: True
```

## 주요 클래스 설명

### HaystackGenerator (건초더미 생성기)
키-값 쌍과 검색 스레드를 생성하는 클래스입니다.

```python
generator = HaystackGenerator(size=50, tokenizer=tokenizer)
haystack = generator.generate()
```

### NeedleThreadingTest (바늘 찾기 테스트)
테스트를 실행하고 결과를 수집하는 메인 클래스입니다.

```python
tester = NeedleThreadingTest()
results = tester.run_tests()
```

## 테스트 종류

### 1. 단일 검색 테스트 (Single Needle Test)
- 주어진 키에 대한 정확한 값을 찾는 테스트
- 모델의 직접적인 키-값 검색 능력을 평가
- 건초더미에서 특정 바늘을 찾는 것과 같은 과정

### 2. 연쇄 검색 테스트 (Threading Test)
- 여러 단계의 연속적인 키-값 검색을 수행
- 찾은 값을 다음 단계의 키로 사용
- 실타래를 따라가듯이 연결된 값들을 찾아가는 과정
- 모델의 연쇄적 추론 능력 평가

## 설정 옵션

```python
NeedleThreadingTest(
    model_name="meta-llama/Llama-3.2-1B",  # 사용할 모델
)

run_tests(
    num_trials=3  # 각 크기별 테스트 반복 횟수
)
```

## 결과 저장

테스트 결과는 다음과 같은 JSON 형식으로 저장됩니다:
```json
{
  "model": "Llama-3.2-1B",
  "timestamp": "2024-01-01T00:00:00",
  "tests": [
    {
      "task": "single_needle",
      "size": 5,
      "trial": 0,
      "duration": 1.23,
      "is_correct": true,
      "expected": "uuid-value",
      "received": "uuid-value"
    }
  ]
}
```

## 라이센스

MIT License

Copyright (c) 2024 

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

## 인용

이 프로젝트를 인용하실 때는 다음 논문을 참조해 주세요:

```bibtex
@article{liu2023needlethreading,
      title={Needle in a Haystack: A Comprehensive Analysis of LLM Behavior in Complex Text Reasoning Tasks}, 
      author={Liu, Wei and Zhang, Zhiwei and Chen, Yiran and Xie, Yuxin and Li, Bo and Shi, Lei},
      journal={arXiv preprint arXiv:2411.05000},
      year={2023}
}
```