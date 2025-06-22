import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.tools import tool
import scipy.stats as stats
import json
from langchain.schema import Document
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.utilities import PythonREPL
from utils import encode_image
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import seaborn as sns 
import matplotlib.pyplot as plt
openai_api_key = st.secrets['OPENAIKEY']
perplexity_key = st.secrets['PERPLEXITY']

# 현재 실행 중인 파일의 디렉토리 절대 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 한 집단 평균검정
@tool # 이것처럼 바꾸기 
def one_mean_test(k:float,col:str,alpha:float):
    """ 
    Test whether the mean of a group of data is statistically k.\
    if pvalue < alpha, select H1\
    col : column name for analyze\
    alpha : 유의수준, not required args\
    Your Annswer:\
    H0 :, H1 :\
    Certification process :\
    Test result :\
    Interpretation of results :\
    """
    data=st.session_state['dataframes'][st.session_state['selected_filename']]
    _, shapiro_p = stats.shapiro(data[col])
    if shapiro_p>=alpha: 
        _,p = stats.ttest_1samp(data[col],k)  
        p=round(p,5)   
    else:
        _,p=stats.wilcoxon(data[col]-k)
    
    if p<alpha:
        select = 'H1'
    else:
        select = 'H0'
    result_info={   
        "pvalue" : p,
        "alpha" : alpha,
        'select' : select
    }
   
    return json.dumps(result_info)

# 두 집단 평균 검정
@tool 
def two_mean_test(col1:str,col2:str,alpha=0.05):
    """
    Test whether the means of the two groups are the same.\
    col1 : first column name for analyze\
    col2 : second column name for analyze\
    alpha : 유의수준, not required args\
    Your Annswer:\
    H0 :, H1 :\
    Certification process :\
    Test result :\
    Interpretation of results :\
    """
    data = st.session_state['dataframes'][st.session_state['selected_filename']]

    _, shapiro_p1 = stats.shapiro(data[col1].values)
    _, shapiro_p2 = stats.shapiro(data[col2].values)
    if shapiro_p1>=alpha and shapiro_p2>=alpha:
        _,p = stats.ttest_ind(data[col1].values, data[col2].value)
        p=round(p,5)
    else:
        _,p=stats.wilcoxon(data[col1],data[col2])
        p=round(p,5)
 
    if p<alpha:
        select = 'H1'
    else:
        select = 'H0'
    
    result_info={
        "pvalue" : p,
        "alpha" : alpha,
        'select' : select,
        f'shapiro pvalue {col1}' : shapiro_p1,
        f'shapiro pvalue {col1}' : shapiro_p1,
    }
    return json.dumps(result_info)

# 이거를 두개로 쪼개기 (분포분석과 가속수명시험으로)
@st.cache_resource # caching 
def make_retrieval_1(): 
    FAISS_INDEX_PATH = os.path.join(current_dir,'faiss_index_ALT')
    # FAISS 인덱스가 이미 존재하는지 확인
    if os.path.exists(FAISS_INDEX_PATH):
        print("디스크에서 FAISS 인덱스를 로드 중...")
        vector = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(api_key=openai_api_key),allow_dangerous_deserialization=True)
    else:
        loader = Docx2txtLoader(os.path.join(current_dir,'가속수명시험.docs'))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = loader.load_and_split(text_splitter)
        
        # FAISS 인덱스를 생성하고 저장
        print("FAISS 인덱스 생성 중...")
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings(api_key=openai_api_key))
        vector.save_local(FAISS_INDEX_PATH)
    
    # Retriever를 생성합니다.
    retrieval = vector.as_retriever()
    retrieval_tool = create_retriever_tool( 
        retrieval,
        name="ALT_search",
        description="It is used to search for content related to accelerated life testing.",
    )
    print('maked RAG1')
    return retrieval_tool

@st.cache_resource # caching 
def make_retrieval_2():
    FAISS_INDEX_PATH = os.path.join(current_dir,'faiss_index_dist')
    # FAISS 인덱스가 이미 존재하는지 확인
    if os.path.exists(FAISS_INDEX_PATH):
        print("디스크에서 FAISS 인덱스를 로드 중...")
        vector = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(api_key=openai_api_key),allow_dangerous_deserialization=True)
    else: 
        loader = Docx2txtLoader(os.path.join(current_dir,'분포분석.docs'))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = loader.load_and_split(text_splitter)

        # FAISS 인덱스를 생성하고 저장
        print("FAISS 인덱스 생성 중...")
        vector = FAISS.from_documents(split_docs, OpenAIEmbeddings(api_key=openai_api_key))
        vector.save_local(FAISS_INDEX_PATH)

    retrieval = vector.as_retriever()
    retrieval_tool = create_retriever_tool( 
        retrieval,
        name="dist_search",
        description="Used when answering questions related to distribution or distribution estimation methods.",
    )
    print('maked RAG2')
    return retrieval_tool

@st.cache_resource
def make_retrieval_from_dataframe(selected_filename):
    df= st.session_state['dataframes'][selected_filename]
    # 데이터프레임을 텍스트로 변환
    df_text = df.to_string(index=False)

    # 텍스트를 문서 형식으로 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_texts = text_splitter.split_text(df_text)

    # 분할된 텍스트를 Document 객체로 변환
    documents = [Document(page_content=text) for text in split_texts]

    # 문서를 VectorStore에 저장
    vector = FAISS.from_documents(documents, OpenAIEmbeddings(api_key=openai_api_key))

    # Retriever 생성
    retrieval = vector.as_retriever()

    retrieval_tool = create_retriever_tool(
        retrieval,
        name="dataframe_search",
        description="Used to provide information about the currently selected data.", 
    )
    return retrieval_tool

@tool
def python_repl(query): 
    """
    A Python shell. Use this to execute python commands. Input should be a valid python command.\
    When visualizing, the image is returned to the following path: os.path.join('plots',img_name.png).\
    and answer is  ![여기](./plots/img_name.png). All img_names(graph) must be different.\
    All graph legends, titles, etc. must be written in English.\
    """
    python_repl = PythonREPL()
    python_repl.locals = python_repl.globals
    return python_repl.run(query)

@tool
def analyze_image(user_request,image_name):
    """Use this tool when interpreting or analyzing images."""
    image_path = f"./plots/{image_name}"
    if os.path.exists(image_path):
        llm = ChatOpenAI(model='gpt-4o',max_retries=5, temperature=0, top_p=0.8, api_key=openai_api_key)
        # 이미지 파일을 base64로 인코딩하고 입력으로 사용
        base64_image = encode_image(image_path)
        image_analysis_input = HumanMessage(
            content=[
                {"type": "text", "text": user_request},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ],
        )
        response = llm.invoke([image_analysis_input])
        return response

@tool
def perplexity(user_request):
    """Use it when searching information on the web that you are not familiar with.You must leave [perplexity] after your answer."""
    chat = ChatPerplexity(temperature=0, pplx_api_key=perplexity_key, model="llama-3.1-sonar-small-128k-online")

    system = "Think about changing Korean to English,but Answer in Korean. be sure to leave [perplexity] at the end of your answer."

    human = "{input}"  
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat 
    response = chain.invoke({"input": user_request})
    
    return response






