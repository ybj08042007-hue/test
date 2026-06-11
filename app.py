import streamlit as st
import google.generativeai as genai
import sympy as sp
import json
from PIL import Image

# ==========================================
# 1. 初始化與 API 設定
# ==========================================
st.set_page_config(page_title="靜力學自由體圖 AI 解題器", layout="wide")
st.title("🏗️ 靜力學自由體圖解題器 (AI 視覺 + SymPy 運算)")
st.markdown("上傳題目圖片，AI 將辨識參數，您可以微調後交由程式計算出**絕對正確**的答案。")

# 請在 Streamlit Secrets 或環境變數中設定您的 API Key
API_KEY = st.text_input("請輸入您的 Google Gemini API Key:", type="password")
if API_KEY:
    genai.configure(api_key=API_KEY)

# ==========================================
# 2. 定義計算引擎 (SymPy 剛體平衡)
# ==========================================
def solve_statics(data):
    """
    接收結構化數據，使用 SymPy 解開 2D 剛體平衡方程式
    """
    # 建立未知數清單與方程式清單
    unknowns = []
    Fx_eq = 0
    Fy_eq = 0
    Moment_eq = 0  # 以原點 x=0 為力矩中心
    
    results_log = []

    try:
        # 處理外加點載重 (Point Loads)
        for load in data.get("point_loads", []):
            mag = load["magnitude"] # 假設向下為負
            x = load["x"]
            Fy_eq -= mag
            Moment_eq -= mag * x
            results_log.append(f"加入點載重: {mag} N 在 x={x} m")

        # 處理均佈載重 (Distributed Loads) - 簡化為矩形或三角形處理
        for d_load in data.get("distributed_loads", []):
            w1, w2 = d_load["start_mag"], d_load["end_mag"]
            x1, x2 = d_load["start_x"], d_load["end_x"]
            length = x2 - x1
            
            # 拆解為矩形與三角形載重
            # 矩形部分
            rect_force = min(w1, w2) * length
            rect_x = x1 + length / 2
            Fy_eq -= rect_force
            Moment_eq -= rect_force * rect_x
            
            # 三角形部分 (假設 w1 > w2)
            if w1 != w2:
                tri_force = abs(w1 - w2) * length / 2
                # 重心位置取決於哪邊比較高
                tri_x = x1 + length / 3 if w1 > w2 else x1 + (2 * length) / 3
                Fy_eq -= tri_force
                Moment_eq -= tri_force * tri_x
                
            results_log.append(f"加入均佈載重: {w1} 到 {w2} N/m，區間 {x1}m~{x2}m")

        # 處理支承反力 (Supports)
        for support in data.get("supports", []):
            name = support["name"]
            stype = support["type"]
            x = support["x"]
            
            if stype == "roller" or stype == "滾子":
                Ry = sp.Symbol(f"{name}_y")
                unknowns.append(Ry)
                Fy_eq += Ry
                Moment_eq += Ry * x
                results_log.append(f"建立滾子支承 {name}: 未知數 {Ry}")
                
            elif stype == "pin" or stype == "插銷":
                Rx = sp.Symbol(f"{name}_x")
                Ry = sp.Symbol(f"{name}_y")
                unknowns.extend([Rx, Ry])
                Fx_eq += Rx
                Fy_eq += Ry
                Moment_eq += Ry * x
                results_log.append(f"建立插銷支承 {name}: 未知數 {Rx}, {Ry}")

        # 解方程式
        equations = [sp.Eq(Fx_eq, 0), sp.Eq(Fy_eq, 0), sp.Eq(Moment_eq, 0)]
        solution = sp.linsolve(equations, unknowns)
        
        return solution, unknowns, results_log, equations
        
    except Exception as e:
        return None, None, [f"計算發生錯誤: {str(e)}"], None

# ==========================================
# 3. AI 圖像辨識與 UI 流程
# ==========================================
uploaded_file = st.file_uploader("上傳靜力學題目 (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and API_KEY:
    image = Image.open(uploaded_file)
    st.image(image, caption="上傳的題目", use_column_width=True)

    if st.button("讓 AI 辨識題目參數"):
        with st.spinner("AI 正在解析圖像..."):
            
        model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = """
            你是一個靜力學專家。請分析這張圖片中的剛體平衡題目（主要為水平樑結構）。
            請將圖片中的資訊提取出來，並**嚴格**以 JSON 格式輸出，不要包含任何其他文字。
            
            JSON 結構範例：
            {
              "supports": [
                {"name": "A", "type": "roller", "x": 0},
                {"name": "B", "type": "pin", "x": 6}
              ],
              "point_loads": [
                {"magnitude": 5000, "x": 2}
              ],
              "distributed_loads": [
                {"start_mag": 900, "end_mag": 600, "start_x": 0, "end_x": 6}
              ]
            }
            備註：長度單位預設為 m，力量預設為 N。起點 x=0 設定在樑的最左端。
            """
            
            try:
                response = model.generate_content([prompt, image])
                # 清理 AI 可能產生的 Markdown 標籤
                json_str = response.text.replace("```json", "").replace("```", "").strip()
                extracted_data = json.loads(json_str)
                st.session_state['extracted_data'] = extracted_data
                st.success("辨識完成！請在下方確認參數是否正確。")
            except Exception as e:
                st.error(f"AI 解析失敗，請手動輸入。錯誤訊息：{e}")

    # 若有提取資料，讓使用者透過 UI 進行最終確認 (Human-in-the-Loop)
    if 'extracted_data' in st.session_state:
        st.subheader("⚙️ 參數確認與微調 (AI 難免出錯，請確認後再計算)")
        
        # 使用 Streamlit 原生輸入框讓使用者修改 JSON
        edited_json = st.text_area("編輯 JSON 參數 (確認數字與圖片相符)", 
                                   value=json.dumps(st.session_state['extracted_data'], indent=4, ensure_ascii=False),
                                   height=300)
        
        if st.button("開始計算支承反力"):
            final_data = json.loads(edited_json)
            solution, unknowns, logs, equations = solve_statics(final_data)
            
            st.divider()
            st.subheader("🧮 計算結果")
            
            for log in logs:
                st.text(log)
                
            st.markdown("### 平衡方程式：")
            st.latex(f"\\sum F_x = 0 \\Rightarrow {sp.latex(equations[0].lhs)} = 0")
            st.latex(f"\\sum F_y = 0 \\Rightarrow {sp.latex(equations[1].lhs)} = 0")
            st.latex(f"\\sum M_O = 0 \\Rightarrow {sp.latex(equations[2].lhs)} = 0")
            
            st.markdown("### 最終解答：")
            if solution:
                sol_list = list(solution)[0]
                for var, val in zip(unknowns, sol_list):
                    st.success(f"**{var} = {val:.2f} (向右/向上為正)**")
            else:
                st.error("方程式無法求解，請檢查輸入參數的自由度是否足夠。")
