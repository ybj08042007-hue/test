import os
import re
import time
import streamlit as st
from google import genai
from PIL import Image

# 1. 網頁基本設定
st.set_page_config(page_title="工程力學解題器", page_icon="📸", layout="centered")

def clean_ai_output(text):
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'​', '', text)
    text = text.replace('$$ ', '$$\n').replace(' $$', '\n$$')
    return text

# 介面標題
st.title("📸 工程力學圖片辨識自動解題器")
st.caption("終極雙保險抗塞車網頁版")
st.divider()

# 2. 設定 API Key (Streamlit 的密碼輸入框)
st.subheader("🔑 步驟 1：請輸入你的 Gemini API Key")
MY_API_KEY = st.text_input("請貼上你的 API Key (輸入內容會自動隱藏)：", type="password")

# 3. 上傳圖片 (Streamlit 的檔案上傳器)
st.subheader("📥 步驟 2：請上傳你的題目圖片")
uploaded_file = st.file_uploader("請選擇或拖曳題目圖片至此...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 讀取並顯示圖片
    img = Image.open(uploaded_file)
    st.image(img, caption="已成功讀取圖片", use_container_width=True)
    
    # 4. 開始解題按鈕
    if st.button("🚀 開始自動解題", type="primary"):
        if not MY_API_KEY.strip():
            st.error("❌ 錯誤：請先在步驟 1 輸入有效的 Gemini API Key！")
        else:
            # 初始化 Gemini Client
            client = genai.Client(api_key=MY_API_KEY.strip())

            prompt = """
            你是一位精通大一工程力學（靜力學）的教授。
            請仔細辨識這張圖片中的題目內容、自由體圖（FBD）、已知的力、距離與支承形式。
            請依據剛體平衡的原理，提供繁體中文的詳細解題步驟。

            【格式限制（極度重要）】：
            1. 嚴禁輸出任何 HTML 標籤（例如 &nbsp;, <br> 等）。
            2. 請使用標準的 LaTeX 格式來表達變數與公式，例如：$A_x$, $B_x$, $B_y$。
            3. 獨立公式請用 $$ 換行包覆，例如：
                $$\\sum F_x = A_x + B_x = 0$$

            請包含以下內容：
            1. 【題目文字辨識】
            2. 【建立座標系與符號定義】
            3. 【列出平衡方程式】
            4. 【聯立求解過程】
            5. 【最終答案與方向說明】
            """

            st.divider()
            
            # 定義備用模型清單
            models_to_try = ['gemini-2.5-flash', 'gemini-2.5-pro']
            success = False

            for model_name in models_to_try:
                if success:
                    break

                status_text = st.empty()
                status_text.info(f"🤖 嘗試使用模型：`{model_name}` 進行解題...")

                max_retries = 2
                retry_delay = 10

                for attempt in range(max_retries):
                    try:
                        # 呼叫 Gemini API
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[img, prompt]
                        )

                        cleaned_text = clean_ai_output(response.text)
                        
                        status_text.empty() # 清除載入中提示
                        st.success("🎉 【自動解題結果】")
                        st.markdown(cleaned_text)
                        success = True
                        break

                    except Exception as e:
                        error_msg = str(e)
                        if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                            st.warning(f"⚠️ {model_name} 伺服器忙碌中 (嘗試第 {attempt + 1}/{max_retries} 次)... 將在 {retry_delay} 秒後重試。")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                        else:
                            st.error(f"發生其他錯誤: {e}")
                            break

                if not success:
                    st.error(f"❌ {model_name} 嘗試失敗，準備切換下一個備援模型...\n")

            if not success:
                st.error("😭 所有模型目前皆處於高負載狀態")
                st.info("💡 **建議解法**：請等待大約 1 到 2 分鐘，直接再次點擊「開始自動解題」按鈕再試一次！")
