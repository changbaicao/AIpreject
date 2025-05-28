import io
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

load_dotenv()


def get_ai_response(memory, user_prompt, system_prompt):
    """根据选择的模型获取AI响应"""
    model = ChatOpenAI(
        model=st.session_state.selected_model,
        base_url='https://twapi.openai-hk.com/v1',
        api_key="hk-mjaz441000055433690fea7d56462f12f3d05d9e764f7c81",
        temperature=st.session_state.model_temperature,
        max_tokens=st.session_state.model_max_length
    )
    chain = ConversationChain(llm=model, memory=memory)
    full_prompt = f"{system_prompt}\n{user_prompt}"
    return chain.invoke({'input': full_prompt})['response']


def extract_chart_type(ai_response):
    chart_keywords = {
        "折线图": ["line", "趋势", "变化"],
        "柱状图": ["bar", "比较", "分布"],
        "饼图": ["pie", "比例", "占比"],
        "散点图": ["scatter", "关系", "相关性"]
    }
    for chart, keywords in chart_keywords.items():
        for keyword in keywords:
            if keyword in ai_response.lower():
                return chart
    return "折线图"


def generate_chart(df, chart_type):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#F8F9FF')
    ax.set_facecolor('#F8F9FF')
    colors = ['#2A27C7', '#6C63FF', '#00B4D8']

    try:
        if "折线图" in chart_type:
            ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=colors[0], marker='o')
        elif "柱状图" in chart_type:
            ax.bar(df.iloc[:, 0], df.iloc[:, 1], color=colors)
        elif "饼图" in chart_type:
            ax.pie(df.iloc[:, 1], labels=df.iloc[:, 0], autopct='%1.1f%%', colors=colors)
        elif "散点图" in chart_type:
            ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color=colors[0])
        else:
            st.error(f"不支持的图表类型：{chart_type}")
            return

        ax.set_title(chart_type, fontsize=14, color='#2A27C7')
        if chart_type != "饼图":
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        st.download_button(
            label="下载图表",
            data=buf.getvalue(),
            file_name=f"superai_chart_{datetime.now().strftime('%Y%m%d%H%M')}.png",
            mime="image/png"
        )
    except Exception as e:
        st.error(f"图表生成失败：{str(e)}")
    finally:
        plt.close()


header_container = st.container()
with header_container:
    cols = st.columns([1, 8, 1])
    with cols[1]:
        st.markdown("""
            <div style="text-align:center; margin-bottom:40px">
                <h1 style="margin-bottom:0">SuperAI 智能分析助手🚀</h1>
                <p style="color:#6C63FF; font-size:1.2rem">数据洞察从未如此简单</p>
            </div>
        """, unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = [{'role': 'ai', 'content': '你好，我是你的AI助手，请问有什么能帮助你吗？'}]
    st.session_state.memory = ConversationBufferMemory(return_messages=True)
    st.session_state.history = []
    st.session_state.df = None
    st.session_state.txt_content = None

with st.sidebar:
    st.title("模型配置")

    st.session_state.selected_model = st.selectbox(
        "选择AI模型",
        ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo", "gpt-4"],
        index=0,
        help="选择要使用的AI模型"
    )

    st.session_state.model_temperature = st.slider("温度 (Temperature)", 0.0, 1.0, 0.7, 0.1)
    st.session_state.model_max_length = st.slider("最大长度", 100, 2000, 1000)
    system_prompt = st.text_area("系统提示词", "你是一个乐于助人的AI助手，用中文回答问题")

    st.subheader("上传数据文件")
    file = st.file_uploader("上传CSV、Excel或TXT文件", type=["csv", "xlsx", "txt"])
    if file:
        try:
            file_type = file.name.split('.')[-1]
            if file_type in ['csv', 'xlsx']:
                df = pd.read_csv(file) if file_type == 'csv' else pd.read_excel(file)
                st.session_state.df = df
                st.dataframe(df.head(8), use_container_width=True, height=300)
                st.caption(f"数据维度：{df.shape[0]} 行 × {df.shape[1]} 列")
            elif file_type == 'txt':
                txt_content = file.read().decode('utf-8')
                st.session_state.txt_content = txt_content
                st.text_area("TXT文件内容", txt_content, height=300)
        except Exception as e:
            st.error(f"数据加载失败：{str(e)}")

    st.subheader("历史消息")
    if st.session_state.messages:
        selected_history = st.selectbox(
            "选择历史对话",
            options=[msg["content"][:30] + "..." if len(msg["content"]) > 30 else msg["content"]
                     for msg in st.session_state.messages if msg["role"] == "human"],
            key="history_select"
        )
        if selected_history:
            for idx, message in enumerate(st.session_state.messages):
                if message["role"] == "human" and message["content"].startswith(selected_history[:30]):
                    st.markdown(f"**用户**: {message['content']}")
                    if idx + 1 < len(st.session_state.messages):
                        st.markdown(f"**AI**: {st.session_state.messages[idx + 1]['content']}")

    if st.button("清空对话历史"):
        st.session_state.messages = [{'role': 'ai', 'content': '你好，我是你的AI助手，请问有什么能帮助你吗？'}]
        st.session_state.memory.clear()

for message in st.session_state.messages:
    role = "user" if message["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.write(message["content"])

if prompt := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({'role': 'human', 'content': prompt})

    with st.spinner('AI 正在思考，请稍等...'):
        progress_bar = st.progress(0)
        response_container = st.empty()
        full_response = ""

        analysis_request = f"分析请求：{prompt}\n"
        if st.session_state.df is not None:
            analysis_request += f"数据结构：{st.session_state.df.columns.tolist()}\n"
            analysis_request += f"数据维度：{st.session_state.df.shape}\n"

            # 根据数据集大小决定是否包含完整数据
            if st.session_state.df.size < 10000:  # 如果数据大小小于10000，包含完整数据
                analysis_request += f"完整数据：{st.session_state.df.to_csv(index=False)}\n"
            else:
                analysis_request += "数据较大，未包含完整数据。\n"
        elif st.session_state.txt_content:
            analysis_request += f"文本内容：{st.session_state.txt_content[:1000]}..."

        ai_response = get_ai_response(st.session_state.memory, analysis_request, system_prompt)

        for i in range(len(ai_response)):
            full_response = ai_response[:i + 1]
            response_container.write(full_response)
            progress_bar.progress((i + 1) / len(ai_response))
            time.sleep(0.03)

        st.session_state.messages.append({'role': 'ai', 'content': full_response})
        st.session_state.memory.save_context({'input': prompt}, {'output': full_response})

        if any(keyword in full_response for keyword in ["图表", "图形", "可视化"]):
            chart_type = extract_chart_type(full_response)
            if st.session_state.df is not None:
                generate_chart(st.session_state.df, chart_type)