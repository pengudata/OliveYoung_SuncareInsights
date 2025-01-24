import streamlit as st
import json
from openai import OpenAI
from pinecone import Pinecone
import base64


api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key = api_key)
pc = Pinecone(api_key = st.secrets["PINECONE_API_KEY"])
index = pc.Index("suncare")

skintype = ["건성", "민감성", "지성", "복합성", "중성", "트러블성"]
# concearn = ["블랙헤드", "미백", "모공", "민감성", "트러블", "각질", "다크서클"]

def get_fullquery(skin_type, query):
    full_query = f"""
유저가 사용하고 싶은 제품을 설명합니다.
이를 참고하여 유저에게 제품을 추천해주세요.
```
피부타입 : {skin_type}
사용하고싶은 제품 : {query}
```
""".strip()
    return full_query


def extract_embedding(text_list):
    response = client.embeddings.create(
        input=text_list,
        model="text-embedding-3-large"
    )
    embedding_list = [x.embedding for x in response.data]
    return embedding_list

def search(query_embedding):
    results = index.query(
        vector=query_embedding,
        top_k=3,
        include_metadata=True,
        filter={
            "skin_type": query_skin or "empty"
        }
    )
    return results

def parse_search_results(results):
    matches = results["matches"]
    metadata_list = [x["metadata"] for x in matches]
    item_list = [{
        "제품이름": x["product_name"],
        "피부타입": x["skin_type"],
        "리뷰": x["review"]
    } for x in metadata_list]
    return item_list

def generate_prompt(query, items):
    prompt = f"""
당신은 올리브영 점장입니다.
유저의 피부타입과 사용하고 싶은 제품에 대한 설명, 이에 대한 추천 결과가 주어집니다.
유저의 입력과 각 추천 결과 제품이름, 피부타입, 리뷰 등을 참고하여 추천사를 작성하세요.
당신에 대한 소개를 먼저 하고, 친절한 말투로 작성해주세요.
중간 중간 이모지를 적절히 사용해주세요.
```
query : {query}
items : {items}
```
    """
    return prompt

def request_chat_completion(prompt):
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = [
            {"role" : "system", "content" : "당신은 제품을 추천해주는 올리브영 AI 점장 곽두팔입니다."},
            {"role" : "user", "content" : prompt}
        ],
        stream = True
    )
    return response

def print_streaming_response(response):
    container = st.empty()
    content = ""
    for chunk in response:
        delta = chunk.choices[0].delta
        if delta.content:
            content += delta.content
            container.markdown(content)


image_path = "./data/oliveyoung.png"

# 이미지를 Base64로 인코딩하여 HTML에 삽입
with open(image_path, "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/jpeg;base64,{b64_string}" alt="Olive Young Logo" style="width:50px; height:50px; margin-right:10px;">
        <h1 style="display:inline; font-size:2em;">올리브영 선케어 제품 추천 사이트</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 제품 추천 헤더 텍스트 줄바꿈 방지 */
    .stForm div:first-child h2 {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-bottom: 20px; /* 아래쪽 여백 추가 */
    }
    
    /* 페이지 상단 문구를 가운데 정렬 */
    div[data-testid="stMarkdownContainer"] p {
        text-align: center;
    }

    /* 제품 추천 텍스트와 이미지를 포함하는 div를 중앙 정렬 */
    div[data-testid="stForm"] > div:first-child {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        text-align: center;
    }

    /* 피부 타입을 선택해주세요 라벨을 중앙 정렬 */
    div[data-testid="stForm"] label {
        display: block;
        text-align: center;
    }

    /* 모든 submit 버튼을 중앙으로 정렬 */
    .stButton button {
        display: block;
        margin: 0 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("./data/suncare.jpg", width=700)
st.markdown(
    """
    <p style="font-size:20px; color:black; font-weight:bold;">
    자신의 피부 유형과 고민거리에 맞게 선케어 제품을 선택하세요!
    </p>
    """,
    unsafe_allow_html=True
)

image = "./data/sunproduct.png"  # 업로드된 이미지 경로
with open(image, "rb") as img_file:
    b64_string = base64.b64encode(img_file.read()).decode()
# st.header 대신 st.markdown을 사용하여 이미지를 포함한 헤더 생성

with st.form("form"):
    st.markdown(
        f"""
        <div class="stForm" style="display:flex; align-items:center;">
            <img src="data:image/png;base64,{b64_string}" alt="Sun Product" style="width:30px; height:30px; margin-right:10px;">
            <h2 style="display:inline; font-size:1.5em;">원하시는 제품 있으시면 말씀해주세요</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    query_skin = st.selectbox(
        label="어떤 피부타입이신가요??",
        options=list(skintype),
        key="query_skin"  # 고유한 키 추가
    )

    query_wanting = st.text_input("원하시는 제품의 특징이 있으신가요??")

    # 폼 제출 버튼
    submitted = st.form_submit_button("추천 받기👍")
    if submitted:
        if len(query_wanting) == 0:
            st.error("원하시는 특징을 설명해주세요.")
        else:
            with st.spinner("관련 제품을 탐색 중입니다..."):
                full_query = get_fullquery(query_skin, query_wanting)
                query_embedding = extract_embedding([
                    full_query
                ])
                results = search(query_embedding[0])
                item_list = parse_search_results(results)
                for item in item_list:
                    with st.expander(item["제품이름"]):
                        st.markdown(f"**피부타입** : {item['피부타입']}")
                        st.markdown(f"**리뷰** : {item['리뷰']}")
            with st.spinner("추천서를 작성 중입니다..."):
                prompt = generate_prompt(
                    query = query_wanting,
                    items = json.dumps(item_list, indent = 2, ensure_ascii = False)
                )
                response = request_chat_completion(prompt)
            print_streaming_response(response)
st.markdown(
    """
    <style>
    /* st.form 전체를 흰색으로 설정 */
    div[data-testid="stForm"] {
        background-color: #C8E6C9; 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1); /* 살짝 그림자 추가 */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 전체 앱 배경색 */
    .stApp {
        background-color: #A7C947; /* 연두색 배경 */
    }
    </style>
    """,
    unsafe_allow_html=True
)
