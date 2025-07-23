# 🧓 Elderly Welfare Recommendation System

노인을 위한 맞춤형 복지정책 추천 시스템입니다.  
공공데이터포털에서 수집한 각 지자체의 정책 데이터를 기반으로 **LLM**과 **RAG 기법**을 활용하여 사용자에게 적절한 복지정책을 안내합니다.

---

## 📌 프로젝트 목적

고령층, 특히 디지털 취약계층에게도 쉽게 접근 가능한 상담형 인터페이스를 제공하여  
복잡하고 흩어져 있는 복지 정책 정보를 쉽고 친절하게 전달하는 것이 목표입니다.

---

## 🔧 구성

elderly_welfare_recommendation_system/
│
├── chat/ # LLM + Gradio 기반 챗봇 인터페이스
│ ├── app.py # 메인 챗봇 실행 파일
│ └── faiss_store/ # 벡터스토어 캐시 디렉토리 (자동 생성됨)
│
├── data/ # 데이터 수집 및 전처리 노트북 및 결과 파일
│ ├── prepare_welfare_data.ipynb
│ ├── processing_welfare_data.ipynb
│ ├── welfare_dataframe.csv # 최종 CSV 데이터
│ └── welfare_data.txt # 원본 XML 데이터
│
├── model/ # 모델 성능 비교 및 RAG 구성
│ ├── llm_evaluation/
│ │ ├── exaone_example.txt
│ │ ├── llama3_example.txt
│ │ ├── evaluation_result.txt
│ │ └── llm_as_a_judge.ipynb
│ └── rag_process.ipynb
│
├── .env # 실제 실행용 환경 변수 (비공개)
├── .env_sample # .env 예시 템플릿
├── requirements.txt # 의존성 패키지 목록
└── README.md # 프로젝트 설명 파일

---

## 🧠 사용한 기술

- **LLM 기반 질의응답**: Exaone (via ChatOllama)
- **RAG (Retrieval-Augmented Generation)**: FAISS + KURE-v1 임베딩
- **데이터 수집**: 공공데이터포털 복지 정책 API 활용
- **챗봇 UI**: Gradio ChatInterface

---

## 📊 LLM 평가 결과

세 가지 모델을 직접 비교하여 최종 선정:

| 모델        | 평가 결과 |
|-------------|------------|
| ✅ Exaone    | 자연스러움, 설명력 우수 |
| LLaMA3      | 정보 부족 |
| EEVE        | 미흡한 답변 |

👉 평가 상세는 `model/llm_evaluation/` 참조

---

## 📎 주요 기능

사용자 질문 → 정책 검색 → LLM 기반 설명

후속 질문(예: “신청 방법은?”, “지원 기간은?”)도 문맥 유지

Gradio UI를 통한 웹 기반 상담 시연 가능

---

## ✨ 향후 개선 아이디어

- 자연어 조건 분석 → 필터링 기반 정책 추천 강화
- 추천 결과 요약 포맷 자동화
- Streamlit 또는 모바일 대응 UI 확장