import os
import re
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image

# 1. 網頁頁面設定
st.set_page_config(
    page_title="工程力學自動解題器",
    page_icon="📸",
    layout="centered"
)

# 清理 AI 輸出的特殊字元
def clean_ai_output(text):
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'​', '', text)
    text = text.replace('$$ ', '$$\n').replace(' $$', '\n$$')
    return text

# 網頁標題
st.title("📸 工程力學圖片辨識自動解題器")
st.markdown("### 🚀 終極雙保險抗塞車網頁版")
st.markdown("---")

# 2. 安全取得 API Key（優先讀取 Secrets，其次讀取側邊欄）
api_key = ""
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
else:
    with st.sidebar:
        st.header("🔑 API 設定")
        api_key = st.text_input("請輸入你的 Gemini API Key", type="password")
        st.markdown("[👉 點此前往 Google AI Studio 申請 Key](https://aistudio.google.com/)")

# 3. 圖片上傳區
st.subheader("📥 步驟 1：請上傳你的題目圖片")
uploaded_file = st.file_uploader("可拖曳或點擊上傳圖片 (支援 PNG, JPG, JPEG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # 讀取圖片並顯示
    img = Image.open(uploaded_file)
    st.image(img, caption="已成功讀取圖片", use_container_width=True)
    
    st.markdown("---")
    
    # 解題按鈕（避免 Streamlit 重新整理時誤觸 API）
    if st.button("🚀 開始自動解題", type="primary"):
        if not api_key.strip():
            st.error("❌ 錯誤：請先在側邊欄輸入你的 Gemini API Key，或於 Secrets 中設定 `GEMINI_API_KEY`！")
        else:
            client = genai.Client(api_key=api_key)

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

            # 定義備用模型清單
            models_to_try = ['gemini-3-flash-preview', 'gemini-2.5-flash']
            success = False

            for model_name in models_to_try:
                if success:
                    break

                status_placeholder = st.empty()
                status_placeholder.info(f"🤖 嘗試使用模型：`{model_name}` 進行解題...")

                max_retries = 2
                retry_delay = 10  

                for attempt in range(max_retries):
                    try:
                        # 畫面上轉圈圈提示
                        with st.spinner(f"🔮 模型 `{model_name}` 正在解題中... (嘗試 {attempt + 1}/{max_retries})"):
                            response = client.models.generate_content(
                                model=model_name,
                                contents=[img, prompt]
                            )

                        cleaned_text = clean_ai_output(response.text)
                        
                        # 清除狀態提示，印出結果
                        status_placeholder.empty()
                        st.success("🎉 【自動解題結果】")
                        st.markdown(cleaned_text)
                        success = True
                        break 

                    except APIError as e:
                        status_code = getattr(e, 'code', None)
                        if status_code in [429, 503] or "UNAVAILABLE" in str(e):
                            st.warning(f"⚠️ `{model_name}` 伺服器忙碌中。將在 {retry_delay} 秒後進行第 {attempt + 1} 次重試...")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                        else:
                            st.error(f"❌ API 發生錯誤 (狀態碼 {status_code}): {e}")
                            break
                    except Exception as e:
                        st.error(f"❌ 發生非預期錯誤: {e}")
                        break

                if not success:
                    st.error(f"❌ `{model_name}` 嘗試失敗，準備切換下一個備援模型...\n")

            if not success:
                st.error("### 😭 所有模型目前皆處於高負載狀態")
                st.info("💡 **建議解法**：請等待大約 1 到 2 分鐘，直接再次點擊「開始自動解題」按鈕重新跑一次即可！")
