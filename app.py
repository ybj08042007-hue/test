import os
import re
import time
import getpass
from google import genai
from google.genai import types
from PIL import Image
from google.colab import files
from IPython.display import display, Markdown

def clean_ai_output(text):
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'\xa0', ' ', text)
    text = re.sub(r'​', '', text)
    text = text.replace('$$ ', '$$\n').replace(' $$', '\n$$')
    return text

def analyze_mechanics_image_v6():
    display(Markdown("## 📸 工程力學圖片辨識自動解題器 (終極雙保險抗塞車版)"))
    display(Markdown("---"))

    # 1. 上傳圖片
    display(Markdown("### 📥 步驟 1：請上傳你的題目圖片"))
    uploaded = files.upload()

    if not uploaded:
        print("未上傳任何檔案。")
        return

    img_path = list(uploaded.keys())[0]
    img = Image.open(img_path)

    display(Markdown("**已成功讀取圖片：**"))
    img.thumbnail((400, 400))
    display(img)

    # 2. 設定 API Key (改為手動安全輸入)
    display(Markdown("### 🔑 步驟 2：請輸入你的 Gemini API Key"))
    MY_API_KEY = getpass.getpass("請貼上你的 API Key (輸入時不會顯示畫面，貼上後按 Enter 即可): ")

    if not MY_API_KEY.strip():
        print("❌ 錯誤：API Key 不能為空，請重新執行程式！")
        return

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

    display(Markdown("---"))

    full_img = Image.open(img_path)

    # 定義備用模型清單
    models_to_try = ['gemini-3-flash-preview', 'gemini-2.5-flash']
    success = False

    for model_name in models_to_try:
        if success:
            break

        display(Markdown(f"### 🤖 嘗試使用模型：`{model_name}` 進行解題..."))

        max_retries = 2
        retry_delay = 10  # 遇到 503 多等幾秒，給伺服器喘息時間

        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[full_img, prompt]
                )

                cleaned_text = clean_ai_output(response.text)
                display(Markdown("### 🎉 【自動解題結果】"))
                display(Markdown(cleaned_text))
                success = True
                break  # 成功就跳出重試迴圈

            except Exception as e:
                error_msg = str(e)
                # 如果是 503 擁擠或 429 額度問題
                if "503" in error_msg or "UNAVAILABLE" in error_msg or "429" in error_msg:
                    print(f" ⚠️ {model_name} 伺服器忙碌中 (嘗試第 {attempt + 1}/{max_retries} 次)... 將在 {retry_delay} 秒後重試。")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                else:
                    print(f"發生其他錯誤: {e}")
                    break

        if not success:
            print(f"❌ {model_name} 嘗試失敗，準備切換下一個備援模型...\n")

    if not success:
        display(Markdown("### 😭 所有模型目前皆處於高負載狀態"))
        display(Markdown("> **建議解法**：請等待大約 1 到 2 分鐘，直接再次點擊 Colab 的執行按鈕重新跑一次即可！"))

# 執行終極版程式
analyze_mechanics_image_v6()
