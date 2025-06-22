import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
import pandas as pd 
import warnings
import dotenv
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
import os
from utils import print_messages,StreamHandler,get_session_history,update_data
from functions import make_retrieval_1,make_retrieval_2, make_retrieval_from_dataframe, one_mean_test,two_mean_test,python_repl,analyze_image,perplexity
from function2 import find_individual_dist,find_best_dist,analyze_AFT,calculate_lifetime_or_test_time
import re
import shutil
import seaborn as sns 
import matplotlib.pyplot as plt
# 사전에 설정한 비밀번호
PASSWORD = st.secrets['PASSWORD']

def check_password():
    def password_entered():
        # 입력된 비밀번호가 올바르면 세션 상태에 'password_correct'를 True로 설정
        if st.session_state["password"] == PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # 보안상 입력된 비밀번호는 삭제
        else:
            st.session_state["password_correct"] = False

    # 세션 상태에 'password_correct'가 없으면 비밀번호 입력 창을 보여줌
    if "password_correct" not in st.session_state:
        # 비밀번호 입력 창
        st.text_input(r"$\textsf{\Large Enter password}$", type="password", on_change=password_entered, key="password")
        return False 
    elif not st.session_state["password_correct"]:
        # 비밀번호가 틀린 경우 다시 입력 요청
        st.error("Incorrect password")
        st.text_input(r"$\textsf{\Large Enter password}$", type="password", on_change=password_entered, key="password")
        return False
    else:
        # 비밀번호가 맞으면 접근 허용
        return True

if check_password():
    # (최초실행) 폴더가 존재하면 내부 비우기  
    if 'reset_plots' not in st.session_state:
        print('Reset')
        st.session_state['reset_plots']=True
        folder_path = './plots'
        if os.path.exists(folder_path):
            # 폴더 내의 모든 파일 및 폴더 삭제
            for filename in os.listdir(folder_path): 
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # 파일 또는 심볼릭 링크 삭제
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)  # 폴더 삭제
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
        else:
            # 폴더가 존재하지 않으면 생성 
            os.makedirs(folder_path)

    # PROMPT 저장 변수
    if 'PROMPT' not in st.session_state:
        st.session_state['PROMPT'] = ChatPromptTemplate.from_messages(
                    [
                        ("system", '''All answers must be written in Korean. you are a data analytics expert. Your job is to analyze the data or visualize it using maplotlib or seaborn without explanation.\
                        When a user requests an analysis, detail the parameter values ​​required to use the tool. Internally convert any given temperature to Kelvin for use, without notifying the user. When printing the formula, you must enter $name as the membership.\
                        All graph names must be different.'''),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ('user',"{input}"),
                        # 대화 기록을 변수로 사용, history가 MessageHistory의 key가 됨
                        MessagesPlaceholder(variable_name='agent_scratchpad')                      
                    ] 
                )

    st.set_page_config(page_title='gpt',page_icon="🦜")
    st.title("Test")

    # 초기 설정
    initial_content="""
    <h5>저는 데이터 분석을 도와주는 agent 입니다.</h5><br>
    <h6>1. 통계적 검정이 가능합니다</h6>
    <h6>2. 데이터 시각화가 가능합니다</h6>
    <h6>3. 분석과 관련된 답변이 가능합니다</h6>
    """

    # 현재 상태를 실시간으로 업데이트
    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{'role':'assistant','content':initial_content}]
       
    # session ID로 사용자를 구분하고, 이전 대화기록을 저장
    if 'store' not in st.session_state:
        st.session_state['store']= dict()
        
    # 데이터 변수 초기화
    if 'dataframes' not in st.session_state:
        # 현재 사용할 데이터프레임 저장
        st.session_state['dataframes']=dict()
        
        initial_df = pd.DataFrame({'None':['데이터를 업로드하세요']})
        
        st.session_state['dataframes']['None']=initial_df
        # 과거 올렸던 파일 비교용
        st.session_state['prev_dataframes']=[]
        # 선택된 파일의 이름
        st.session_state['selected_filename'] = "None"
        # 이전에 선택된 파일이름 (선택된게 달라지면 업데이트하도록)
        st.session_state['prev_selected_filename'] = "None"

    # 데이터베이스 검색도구 (최초 한번만 생성되도록 설정)
    if 'DB_Retrieval' not in st.session_state:
        st.session_state['DB_Retrieval_1'] = make_retrieval_1()
        st.session_state['DB_Retrieval_2'] = make_retrieval_2()

    if 'uploaded_files' not in st.session_state:
        st.session_state['uploaded_files']=[]
    if 'uploaded_image' not in st.session_state:
        st.session_state['uploaded_image'] = None

    if 'plot' not in st.session_state:
        st.session_state['plot']=[] 

    # 부가적인 기능 처리
    with st.sidebar: 
        session_id =st.text_input('Session ID',value='abc123')

        clear_btn =st.button('세션기록 초기화',type="primary")
        
        # 전체 초기화 (retrieval등 캐쉬상태는 유지)
        if clear_btn:
            # 세션 상태 초기화
            for key in st.session_state.keys():
                # 에외항목 (로그인상태)
                if key != "password_correct":
                    del st.session_state[key]
            # 기본기록 추가 
            st.session_state['messages'] = [] # st의 대화기록 초기화  
            st.session_state['messages'].append({'role':'assistant','content':initial_content})
            #st.session_state['store'] = dict() # store의 대화기록 초기화
            st.rerun() 
        
        # 파일업로드 
        # 파일 업로더를 배치할 빈 공간 생성 
        file_uploader_placeholder = st.empty()

        if 'show_uploader' not in st.session_state:
            st.session_state['show_uploader'] = True  # 초기에는 파일 업로더 표시

        if st.session_state['show_uploader']:
            st.session_state['uploaded_files'] = file_uploader_placeholder.file_uploader("Upload Files", type=["csv", "xls", "xlsx"], accept_multiple_files=True)

            if st.session_state['uploaded_files']: 
                # 파일 업로드 후 버튼 숨기고 파일 업로더 제거 
                st.session_state['show_uploader'] = False
                file_uploader_placeholder.empty()

        # 파일 추가 버튼 표시
        if not st.session_state['show_uploader']:
            if st.button('파일 추가'):
                st.session_state['show_uploader'] = True  # 파일 업로더 다시 표시
                #st.experimental_rerun()  # 화면 업데이트
        # 데이터 최신화 함수
        update_data()
        
        # 현재 분석에 사용할 파일 고르기
        selected_file = st.selectbox('selected file', options= st.session_state['dataframes'])
       
        if selected_file != None: 
            st.session_state['selected_filename']=selected_file
            
            # 선택한 파일이 바꼇을때만 데이터분석도구를 재생성함
            DF_Retrieval = make_retrieval_from_dataframe(st.session_state['selected_filename'])
            
            # 현재 선택이 이전 선태과 다를때 화면상애 표시
            if st.session_state['selected_filename'] != st.session_state['prev_selected_filename']:
                st.session_state['prev_selected_filename']=st.session_state['selected_filename']
                st.session_state['messages'].append({'role':'df', 'content':st.session_state['selected_filename']})

    # 이전 메시지 출력 함수
    print_messages()

    # 이미지 업로드
    img_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
    if img_file is not None:
        os.makedirs("plots", exist_ok=True)
        img_path = os.path.join("plots", img_file.name)
        with open(img_path, "wb") as f:
            f.write(img_file.getbuffer())
        st.session_state['uploaded_image'] = img_path
        st.image(img_path, caption=os.path.basename(img_path))

    user_input = st.chat_input('메시지 입력', key='input')
    if user_input or st.session_state.get('uploaded_image'):
        if user_input:
            st.chat_message('user').markdown(user_input)
            st.session_state['messages'].append({'role':'user','content':user_input})
        if st.session_state.get('uploaded_image'):
            st.image(st.session_state['uploaded_image'], caption=os.path.basename(st.session_state['uploaded_image']))
            st.session_state['messages'].append({'role':'image', 'content': st.session_state['uploaded_image']})
        
        # 그래프 공간 비워두기
        st.session_state['placeholder_plot'] = st.empty()
        
        with st.chat_message('assistant'):
            placeholder = st.empty()
            placeholder.markdown("답변 생성 중...")  # 로딩 메시지
            stream_handler =StreamHandler(placeholder)
            
        
        # 모델 생성
        model=ChatOpenAI(streaming=True, callbacks=[stream_handler],model='gpt-4.1',temperature=0.03, max_retries=7,top_p=0.7,api_key=st.secrets['OPENAIKEY']) #

        # AgentExecutor 클래스를 사용하여 agent와 tools를 설정하고, 상세한 로그를 출력하도록 verbose를 True로 설정합니다.
        tools = [st.session_state['DB_Retrieval_1'],st.session_state['DB_Retrieval_2'],DF_Retrieval, one_mean_test, two_mean_test, python_repl, find_individual_dist,find_best_dist,analyze_AFT,calculate_lifetime_or_test_time,analyze_image,perplexity] # python_repl

        agent = create_openai_functions_agent(model, tools, st.session_state['PROMPT'])

        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

        agent_with_chat_history = RunnableWithMessageHistory(
                agent_executor,
                get_session_history,
                input_messages_key='input',
                history_messages_key='chat_history'
        )
        # 답변처리
        if st.session_state.get('uploaded_image'):
            image_name = os.path.basename(st.session_state['uploaded_image'])
            response = analyze_image(user_input if user_input else '이미지를 분석해줘', image_name)
            msg = response.content if hasattr(response, 'content') else response
            st.session_state['uploaded_image'] = None
        else:
            response = agent_with_chat_history.invoke(
                {'input': user_input},
                config={'configurable': {"session_id": session_id}}
            )

            msg = response['output']
        # 그래프 생성되면, session에 저장하는부분(도구로 처리할까?)
        if "[여기]" in msg and '.png' in msg:
            match = re.search(r'/([^/]+)\.png', msg)
            if match:
                filename = match.group(1)
                filepath=os.path.join('plots',f"{filename}.png")
                st.session_state['messages'].append({'role':'plot', 'content':filepath})
                st.rerun()  # 앱 다시 실행하여 이미지를 화면상에 표시
        st.session_state['messages'].append({'role':'assistant','content':msg}) 
 
