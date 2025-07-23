import os
import gradio as gr
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOllama

# 벡터스토어 캐싱 로직
FAISS_INDEX_PATH = "faiss_store"
DATA_PATH = "../data/welfare_dataframe.csv"

embeddings = HuggingFaceEmbeddings(
    model_name="nlpai-lab/KURE-v1",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

welfare_data = pd.read_csv(DATA_PATH)

def make_policy_chunk(row):
    return f"""[정책명] {row['servNm']}
[지원유형] {row['지원유형']}
[지원내용] {row['alwServCn']}
[대상자 조건] {row['sprtTrgtCn']}
[선정기준] {row['slctCritCn']}
[시행기간] {row['시행기간_표기'] if pd.notna(row['시행기간_표기']) else '무기한'}
[신청안내] {row['신청안내'] if pd.notna(row['신청안내']) else '별도 안내 없음'}
[상세보기] {row['servDtlLink'] if pd.notna(row['servDtlLink']) else '링크 없음'}
"""


if "rag_청크" not in welfare_data.columns:
    welfare_data["rag_청크"] = welfare_data.apply(make_policy_chunk, axis=1)

if os.path.exists(FAISS_INDEX_PATH):
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
else:
    documents = [
        Document(
            page_content="passage: " + row["rag_청크"],
            metadata={"servId": row["servId"], "servNm": row["servNm"]}
        )
        for _, row in welfare_data.iterrows()
    ]
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH)

# 설명 프롬프트와 체인
explanation_template = PromptTemplate.from_template("""
사용자의 질문: "{question}"

다음은 추천된 정책 목록입니다:

{policy}

위 정보들을 바탕으로 다음 지침에 따라 답해주세요:

- 각 정책은 반드시 아래 항목들을 포함해야 합니다:
  - 정책명
  - 신청안내
  - 시행기간
  - 상세보기 링크 (링크는 괄호로 감싸서 표기)
- 예시:
  [정책명] 저소득 노인 건강보험료 지원  
  • 신청안내: 주민센터 방문 또는 홈페이지 신청  
  • 시행기간: 무기한  
  • 상세보기: [바로가기](https://example.com)
- 노년층도 이해할 수 있도록 친절한 말투로 설명
- 줄바꿈으로 항목별 정보가 명확하게 구분되도록 구성
""")


llm_EXAONE = ChatOllama(model="exaone-3.0-7.8b-it-Q8_0:latest")
explanation_chain = explanation_template | llm_EXAONE

# 문맥형 쿼리 빌더
chat_history = []

def contextual_query_builder(history, current_question):
    context = ""
    for i, (q, a) in enumerate(history[-3:]):
        context += f"[이전 질문 {i+1}] {q}\n[이전 답변 {i+1}] {a}\n"
    context += f"[현재 질문] {current_question}"
    return f"query: {context}"

# Gradio 응답 함수
def respond(message, history):
    # 사용자의 현재 질문
    current_question = message

    # 문맥형 질의 생성
    contextual_query = contextual_query_builder(history, current_question)

    # 정책 추천 (항상 3개까지)
    search_results = vectorstore.similarity_search(contextual_query, k=3)

    if not search_results:
        return "죄송해요, 관련된 정책을 찾을 수 없었어요. 다시 질문해 주세요!"

    # 이전 대화 내용을 기반으로 후속 질문인지 확인
    is_follow_up = len(history) > 0 and not any(kw in current_question for kw in ["어떤", "무슨", "추천", "알려줘", "있어"])

    if is_follow_up:
        # 이전 응답 기반으로 후속 질문 처리
        summarized_policies = []
        for doc in search_results:
            md = doc.metadata
            content = doc.page_content
            # 필수 정보 추출
            servNm = md.get("servNm", "정책명 미상")
            lines = content.split("\n")
            시행기간 = next((l.replace("[시행기간]", "").strip() for l in lines if "[시행기간]" in l), "안내되지 않음")
            신청안내 = next((l.replace("[신청안내]", "").strip() for l in lines if "[신청안내]" in l), "안내되지 않음")

            summarized_policies.append(
                f"[정책명] {servNm}\n[시행기간] {시행기간}\n[신청안내] {신청안내}"
            )

        policies_text = "\n\n".join(summarized_policies)

        prompt = PromptTemplate.from_template("""
            사용자의 질문: "{question}"

            다음은 추천된 정책 목록입니다:

            {policy}

            위 정보들을 바탕으로 다음 지침에 따라 답해주세요:

            - 각 정책명은 반드시 포함
            - 신청안내 및 시행기간은 명확히 표시 (없으면 '별도 안내 없음')
            - 노년층이 이해하기 쉬운 말투와 문장 구조로 설명
            - 줄바꿈을 이용해 항목별로 보기 쉽게 정리
            """)

        llm_chain = prompt | llm_EXAONE
        response = llm_chain.invoke({
            "question": current_question,
            "policy": policies_text
        })

    else:
        # 일반 추천 흐름 (이전과 동일)
        policies_text = "\n\n".join([
            f"- {doc.metadata['servNm']}\n{doc.page_content}" for doc in search_results
        ])

        response = explanation_chain.invoke({
            "question": current_question,
            "policy": policies_text
        })

    # AIMessage 대응
    if hasattr(response, "content"):
        response = response.content

    return response


# Gradio UI
chat_interface = gr.ChatInterface(
    fn=respond,
    chatbot=gr.Chatbot(label="노인복지 상담봇"),
    title="노인복지 상담 도우미",
    description="노인복지 관련 정책을 쉽게 안내받을 수 있어요. 궁금한 점을 편하게 물어보세요!",
    theme="soft",
    examples=[
        "서울에 사는 70세 노인이 받을 수 있는 의료지원 정책은 뭐야?",
        "경기도에 거주하는 저소득층 노인을 위한 주거지원 정책 알려줘",
        "부산 사는 65세 여성을 위한 복지 혜택이 뭐야?"
    ],
    submit_btn="질문하기"
)

chat_interface.launch()

