# 📌 작업 지시(Instructions)

## 0. 배경
- 우리는 이미 **텍스트 기반** 멀티턴 챗봇(LLM) 코드를 Streamlit으로 운영 중이다.  
- 현재 **이미지 입력**은 막혀 있다.  
- 목표:  
  1. 사용자가 이미지(.jpg · .png) 파일을 업로드 가능하도록 하고  
  2. 업로드 직후 화면에 표시하며  
  3. 선택적으로 텍스트도 함께 보내면 **멀티모달**로 LLM에 전송  
  4. Streamlit의 *새로고침 문제*를 `st.session_state` 로 해결해 대화·이미지를 **지속(persist)**  

## 1. 수정 지침
### 1-1. 기존 코드 맥락
- `st.text_input` 또는 `st.chat_input` 로 텍스트를 받고 `chat_history` 를 `st.session_state` 에 보관하는 구조다.  
- **중요**: 기존 로직·스타일은 *절대 그대로* 유지 → **필요한 부분만 덧붙여라**.

### 1-2. 추가 기능 요구
1. **이미지 업로더**
   ```python
   img_file = st.file_uploader("이미지 업로드", type=["jpg", "jpeg", "png"])
