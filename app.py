import streamlit as st
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import predict
import numpy as np
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
plt.rcParams['axes.unicode_minus'] = False

model = predict.EmotionInference(model_path='best_model.pth', vocab_path='vocab.json', config_path='config.json')

# ------------------------------
# 页面基本配置 
# ------------------------------
st.set_page_config(
    page_title="nju红黑榜情感分析",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f9f0ff; }
    h1, h2, h3, h4 { color: #6A1B9A; }
    .stButton>button {
        background-color: #8E24AA; color: white; border-radius: 8px; border: none; padding: 0.5em 1.5em;
    }
    .stButton>button:hover { background-color: #7B1FA2; color: white; }
    .stMetric { background-color: #f3e5f5; border-radius: 10px; padding: 1em; }
    div[data-baseweb="input"] > div { border-color: #CE93D8 !important; }
    div[data-baseweb="input"] > div:focus-within { border-color: #8E24AA !important; box-shadow: 0 0 0 1px #8E24AA !important; }
    div[data-baseweb="select"] > div { border-color: #CE93D8 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 数据加载与聚合

def load_teacher_data(json_path="merged_data.json"):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        st.error(f"文件 {json_path} 未找到，请确认路径是否正确。")
        return []
    except json.JSONDecodeError:
        st.error("JSON 文件格式错误，请检查。")
        return []

    teacher_dict = defaultdict(lambda: {"courses": set(), "reviews": [], "details": []})
    for entry in raw_data:
        teacher = (entry.get("教师") or "").strip()
        course = (entry.get("课程名称") or "").strip()
        review = (entry.get("评价_0") or "").strip()

        if not teacher:
            continue
        if course:
            teacher_dict[teacher]["courses"].add(course)
        if review:
            teacher_dict[teacher]["reviews"].append(review)
            teacher_dict[teacher]["details"].append({"course": course, "review": review})

    result = []
    for teacher, info in teacher_dict.items():
        result.append({
            "name": teacher,
            "courses": sorted(list(info["courses"])),
            "reviews": info["reviews"],
            "review_details": info["details"]
        })
    result.sort(key=lambda x: x["name"])
    return result

# 情感分析函数

EMOTION_LABELS = [
    "angry", "fear", "happy", "neutral", "sad", "suprise"
]

def analyze_sentiment(text):
    probs=model.predict(text)
    new_probs=probs.copy()
    indices=[0,1,2,4,5]
    max_val=-np.inf
    max_idx=-1
    for i in indices:
        if probs[i]>max_val:
            max_val=probs[i]
            max_idx=i
    new_probs[max_idx]=max_val*2
    total=np.sum(new_probs)
    if total!=0:
        new_probs=new_probs/total
    return new_probs


def compute_overall_sentiment(reviews):
    if not reviews:
        return 0.0, [0.0] * 6

    distributions = []
    for rev in reviews:
        dist = analyze_sentiment(rev)
        distributions.append(dist)

    # 计算平均分布
    n = len(distributions)
    avg_dist = [sum(d[i] for d in distributions) / n for i in range(6)]

    # 综合得分：示例为积极情绪（索引0和1）的总和
    overall_score = avg_dist[2]*2
    return overall_score, avg_dist


def generate_sentiment_distribution_chart(distribution):
    """
    根据平均情感分布生成环形图（饼图）。
    优化点：使用图例代替外部标签，避免小扇区文字重叠。
    """
    # 稍微增大画布尺寸，为图例留出空间
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["#F94144", "#F9844A", "#F9C74F", "#E9ECEF", "#4D9DE0", "#F72585"]
    
    wedges, texts, autotexts = ax.pie(
        distribution,
        labels=None,                     
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4, edgecolor='white')
    )
    
    # 调整百分比文字样式（黑色更清晰，字体适中）
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_color("black")
    
    # 添加图例，放在饼图右侧，避免遮挡
    ax.legend(wedges, EMOTION_LABELS,
              title="情感类型",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))   # 将图例置于饼图右侧外
    
    ax.set_title("情感分布", fontsize=14, color="#6A1B9A", pad=15)
    fig.patch.set_facecolor('#f9f0ff')
    return fig


# 教师姓名模糊匹配

def find_matching_teachers(query, teachers):
    query = query.strip().lower()
    if not query:
        return []
    matches = []
    seen_names = set()
    for t in teachers:
        name = t.get("name", "")
        if query in name.lower() and name not in seen_names:
            matches.append(t)
            seen_names.add(name)
    return matches

# Streamlit 前端界面

def main():
    st.title("🎓 nju红黑榜情感分析 测试版")
    st.write("注意，本页面暂时无法实现数据的动态拉取，数据从nju.ys.al处获得\n")
    st.write("暂时还无法区分重名教师\n")
    st.write("情感计算部分，采用lstm训练出来的模型")
    st.markdown("### 注意，本页面仅供娱乐！！！不含作者主观倾向")
    st.markdown("—— 基于学生评价的情感指数与分布 ——")
    st.markdown("---")

    teachers = load_teacher_data("merged_data.json")
    if not teachers:
        st.warning("教师数据为空，请确认 JSON 文件路径及内容。")
        st.stop()

    col1, col2 = st.columns([3, 2])
    with col1:
        query = st.text_input("🔍 输入教师姓名（支持模糊匹配）", placeholder="例如：张、李华...")

    matched_teachers = find_matching_teachers(query, teachers) if query else []

    with col2:
        if query:
            if matched_teachers:
                options = []
                for t in matched_teachers:
                    courses = t.get('courses', [])
                    courses_str = ', '.join(courses) if courses else '未知课程'
                    options.append(f"{t['name']}  ({courses_str})")
                selected_label = st.selectbox("请选择你要查看的教师", options)
                selected_name = selected_label.split("  (")[0]
                selected_teacher = next(
                    (t for t in matched_teachers if t["name"] == selected_name), None
                )
            else:
                st.info("未找到匹配的教师，请调整关键词。")
                selected_teacher = None
        else:
            st.info("请输入姓名进行搜索。")
            selected_teacher = None

    if selected_teacher:
        st.markdown("---")
        st.subheader(f"👨‍🏫 {selected_teacher['name']}")
        courses = selected_teacher.get('courses', [])
        st.caption(f"教授课程：{', '.join(courses) if courses else '未提供'}")

        reviews = selected_teacher.get("reviews", [])
        review_details = selected_teacher.get("review_details", [])
        if not reviews:
            st.warning("该教师暂无评价数据。")
        else:
            st.markdown("#### 📋 所有学生评价")
            with st.expander(f"展开查看全部 {len(reviews)} 条评价", expanded=False):
                for i, detail in enumerate(review_details, 1):
                    course = detail['course'] if detail['course'] else '未知课程'
                    st.write(f"{i}. 【{course}】{detail['review']}")

            overall_score, avg_distribution = compute_overall_sentiment(reviews)
            st.markdown("#### 📊 综合情感指数")
            st.metric(label="情感得分（积极倾向）", value=f"{overall_score:.2f}")

            st.markdown("#### 📈 情感分布图表")
            col_left, col_center, col_right = st.columns([1, 2, 1])
            with col_center:
                fig = generate_sentiment_distribution_chart(avg_distribution)
                st.pyplot(fig, use_container_width=False)   

    st.markdown("---")
    st.caption("nju红黑榜情感评价系统 | 数据仅供参考")


if __name__ == "__main__":
    main()