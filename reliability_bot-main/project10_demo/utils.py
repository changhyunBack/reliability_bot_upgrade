import streamlit as  st
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import os
import pandas as pd 
import base64
current_dir = os.path.dirname(os.path.abspath(__file__))

class StreamHandler(BaseCallbackHandler):
    def __init__(self,container,initial_text=""):
        self.container= container
        self.text= initial_text
        self.state='Idle'
    
    def on_llm_new_token(self,token:str, **kwargs) -> None: 
        self.text += token
        self.container.markdown(self.text)
    
def print_messages():
    # 이전 대화기록 불러오기
    if 'messages' in st.session_state and len(st.session_state['messages'])>0:
        for msg in st.session_state['messages']:
            if msg['role'] in ['plot', 'image']:
                st.image(msg['content'], caption=os.path.basename(msg['content']))
            elif msg['role'] =='df':
                if msg['content'] != 'None':
                    st.dataframe(st.session_state['dataframes'][msg['content']]) # 데이터프레임파일 (dict형태로되있음.)
            elif msg['role']=='df_result':
                st.dataframe(msg['content'])
            else:
                st.chat_message(msg['role']).markdown(msg['content'],unsafe_allow_html=True)
 
    
# 세선 ID를 기반으로 세션 기록을 가져오는 함수 (사용자별 대화기록)
def get_session_history(session_ids : str) -> BaseChatMessageHistory:
    # print(session_ids)
    if session_ids not in  st.session_state['store']:
         st.session_state['store'][session_ids] =ChatMessageHistory()
    return  st.session_state['store'][session_ids]


# 데이터 업데이트 
def update_data():
    # 업로드한 파일이 있을때, 읽어들임. 파일이 바뀌면 초기화후 다시실행
    if st.session_state['uploaded_files'] and st.session_state['uploaded_files'] != st.session_state['prev_dataframes']:
        # 과거기록 업데이트
        st.session_state['prev_dataframes']= st.session_state['uploaded_files']
        
        # 기존데이터와 비교해서 새로 추가된걸 저장소에 추가 
        for uploaded_file in st.session_state['uploaded_files']: 
            if uploaded_file.name not in st.session_state['dataframes']:
                if uploaded_file.type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                # key : value로 데이터 저장
                st.session_state['dataframes'][uploaded_file.name] = df

    
def save_plot(fig, filename):
    # 폴더가 없으면 생성
    os.makedirs("plots", exist_ok=True)
    filepath = os.path.join("plots", filename)
    fig.savefig(filepath)
    return filepath

# 이미지 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 실제 이미지 경로를 반환하는 함수
def get_image_path(image_name, base_dir=os.path.join(current_dir,"plots")):
    return os.path.join(base_dir, image_name)


