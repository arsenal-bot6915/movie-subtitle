import math
import re
import smtplib
import time
import concurrent.futures
import threading
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from email.message import EmailMessage
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

try:
    import tomllib  # py3.11+
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore

CACHE_DIR = Path(".srt_cache")
CACHE_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """你是一个专业的影视字幕翻译专家。你需要将收到的英文字幕片段翻译成简体中文字幕。

【核心原则】
1. 严禁合并或拆分字幕：你必须对每个数字ID对应的字幕进行逐一翻译。
2. 术语一致性：严格遵守给定的影片背景和术语表，人名、地名等专有名词保持连贯。
3. 翻译风格：口语化、简练，符合电影对白“短平快”的节奏。单行长度适中。

【强制输出格式】
你必须且只能返回一段合法的 JSON 数据（不需要时间轴，程序会在外部自动合成）。
键(Key)为字幕的原数字ID，值(Value)为翻译后的中文文本。绝对不要包含任何其它说明文本。

【输出范例】
{
  "1": "开什么玩笑？",
  "2": "我们没时间了！快走！"
}"""

DEFAULT_BATCH_SIZE = 80
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 2
DEFAULT_GAP_THRESHOLD_MS = 15000


@dataclass
class SubtitleEntry:
    index: int
    timeline: str
    text: str


def get_cache_file_path(file_content: bytes, model: str, movie_bg: str, glossary: str = "", batch_size: int = 80, gap_ms: int = 15000) -> Path:
    hasher = hashlib.md5()
    hasher.update(file_content)
    hasher.update(model.encode('utf-8'))
    hasher.update(movie_bg.encode('utf-8'))
    hasher.update(glossary.encode('utf-8'))
    hasher.update(str(batch_size).encode('utf-8')) # [新增] 将批次参数加入哈希
    hasher.update(str(gap_ms).encode('utf-8'))     # [新增] 将停顿参数加入哈希
    hasher.update(b"v3_smart_chunk") # [修改] 升级一下版本防冲突
    return CACHE_DIR / f"{hasher.hexdigest()}.json"

def save_progress_to_local(cache_file: Path, translated_dict: Dict[int, List[SubtitleEntry]]):
    data_to_save = {
        str(k): [asdict(entry) for entry in v] 
        for k, v in translated_dict.items()
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False)

def load_progress_from_local(cache_file: Path) -> Dict[int, List[SubtitleEntry]]:
    if not cache_file.exists():
        return {}
    try:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            int(k): [SubtitleEntry(**entry) for entry in v] 
            for k, v in data.items()
        }
    except Exception:
        return {}

def parse_srt(content: str) -> List[SubtitleEntry]:
    entries: List[SubtitleEntry] = []
    lines = content.splitlines()
    
    state = "INDEX"
    current_index = -1
    current_timeline = ""
    current_text = []
    
    for line in lines:
        line = line.strip()
        if not line:
            if state == "TEXT" and current_text:
                entries.append(SubtitleEntry(current_index, current_timeline, "\n".join(current_text)))
                state = "INDEX"
                current_text = []
            continue
            
        if state == "INDEX" and line.isdigit():
            current_index = int(line)
            state = "TIMELINE"
        elif state == "TIMELINE" and "-->" in line:
            current_timeline = line
            state = "TEXT"
        elif state == "TEXT":
            current_text.append(line)
        else:
            if "-->" in line:
                current_index = len(entries) + 1
                current_timeline = line
                state = "TEXT"
                current_text = []

    if state == "TEXT" and current_text:
        entries.append(SubtitleEntry(current_index, current_timeline, "\n".join(current_text)))

    if not entries:
        raise ValueError("未能解析到有效 SRT 内容，请检查文件格式。")
        
    return entries

def entries_to_srt(entries: List[SubtitleEntry]) -> str:
    chunks = []
    for e in entries:
        chunks.append(f"{e.index}\n{e.timeline}\n{e.text}".strip())
    return "\n\n".join(chunks) + "\n"


# [新增] 阶段四核心：计算时间差，解析 SRT 时间戳
def get_srt_time_gap_ms(timeline1: str, timeline2: str) -> int:
    try:
        def to_ms(t_str: str) -> int:
            t_str = t_str.strip()
            h, m, s_ms = t_str.split(':')
            s, ms = s_ms.split(',')
            return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
        
        end_t1 = timeline1.split('-->')[1].strip()
        start_t2 = timeline2.split('-->')[0].strip()
        return to_ms(start_t2) - to_ms(end_t1)
    except Exception:
        return 0

# [新增] 阶段四核心：基于时间停顿智能分批
def smart_chunk_entries(entries: List[SubtitleEntry], max_batch_size: int = DEFAULT_BATCH_SIZE, gap_threshold_ms: int = DEFAULT_GAP_THRESHOLD_MS) -> List[List[SubtitleEntry]]:
    if not entries:
        return []
    batches = []
    current_batch = [entries[0]]
    
    for i in range(1, len(entries)):
        prev_entry = entries[i-1]
        curr_entry = entries[i]
        
        # 计算两条字幕间的时间空隙
        gap = get_srt_time_gap_ms(prev_entry.timeline, curr_entry.timeline)
        
        # 满足以下任意条件即截断：超过最大条数，或者超过设定的时间停顿（换场）
        if len(current_batch) >= max_batch_size or gap >= gap_threshold_ms:
            batches.append(current_batch)
            current_batch = [curr_entry]
        else:
            current_batch.append(curr_entry)
            
    if current_batch:
        batches.append(current_batch)
    return batches


# [新增] 阶段二核心：预处理自动提取术语表
def auto_extract_glossary(client: OpenAI, model: str, entries: List[SubtitleEntry]) -> str:
    # 抽取前 150 条对白，通常包含了主要角色的出场和基本设定
    sample_entries = entries[:150]
    sample_text = "\n".join([e.text for e in sample_entries])
    
    sys_prompt = "你是一个专业影视本地化翻译。你的任务是从提供的英文字幕中提取高频、重要的人名、地名、虚构组织或特定名词，并给出推荐的中立中文翻译。"
    user_prompt = f"【字幕片段】\n{sample_text}\n\n请以 JSON 格式输出，格式如：{{\"John\": \"约翰\", \"FBI\": \"联邦调查局\"}}。请提取出最具代表性的（最多15个以内）。"
    
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=2000,  # <--- [新增] 防止幻觉导致疯狂计费
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"} if model == "deepseek-chat" else None
        )
        content = resp.choices[0].message.content or ""
        match = re.search(r'\{[\s\S]*\}', content)
        if match:
            res_dict = json.loads(match.group(0))
            return "\n".join([f"{k}: {v}" for k, v in res_dict.items()])
        return ""
    except Exception as e:
        return f"AI 提取失败，请手动填写。原因: {e}"


def build_user_prompt(
    batch_entries: List[SubtitleEntry], 
    prev_entries_orig: Optional[List[SubtitleEntry]] = None,
    prev_entries_trans: Optional[List[SubtitleEntry]] = None,
    movie_name: str = "",
    movie_bg: str = "",
    glossary: str = ""
) -> str:
    prompt = ""
    
    if movie_name or movie_bg or glossary:
        prompt += "【影片背景与术语设定】\n"
        if movie_name: prompt += f"- 影视名称：{movie_name}\n"
        if movie_bg: prompt += f"- 剧情/风格提示：{movie_bg}\n"
        if glossary: prompt += f"- 专有名词对照表（务必遵守以下翻译）：\n{glossary}\n\n"

    # [优化] 滑动窗口：强调这是“上一幕”或“上一句”的情境
    if prev_entries_orig and prev_entries_trans and len(prev_entries_orig) == len(prev_entries_trans):
        prompt += "【上文语境参考】（以下是对白的前情提要，仅供你了解对话连贯性和语气，不要翻译它们）：\n"
        for orig, trans in zip(prev_entries_orig, prev_entries_trans):
            prompt += f"原文: {orig.text}\n译文: {trans.text}\n---\n"

    input_dict = {str(e.index): e.text for e in batch_entries}
    prompt += f"\n【待翻译字幕（请仅以 JSON 格式输出这 {len(batch_entries)} 条字幕的翻译结果）】\n"
    prompt += json.dumps(input_dict, ensure_ascii=False, separators=(',', ':'))
    
    return prompt

def extract_and_validate_json(original_batch: List[SubtitleEntry], raw_text: str, strict: bool = True) -> tuple[List[SubtitleEntry], List[int]]:
    text = raw_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    match = re.search(r'\{[\s\S]*\}', text)
    if not match:
        raise ValueError("未能从模型返回内容中提取到有效的 JSON 格式数据。")
        
    try:
        translated_dict = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"模型返回的 JSON 格式损坏无法解析: {e}")

    merged: List[SubtitleEntry] = []
    missing_ids: List[int] = []  # 记录漏翻的 ID
    
    for src in original_batch:
        key = str(src.index)
        if key not in translated_dict:
            missing_ids.append(src.index)
            trans_text = src.text  # 漏翻时：使用英文原文兜底
        else:
            trans_text = str(translated_dict[key]).strip()
            if not trans_text:
                trans_text = src.text
            
        merged.append(SubtitleEntry(index=src.index, timeline=src.timeline, text=trans_text))

    # 严格模式下，有漏翻就报错触发重试
    if strict and missing_ids:
        raise ValueError(f"AI 漏翻了序号为 {missing_ids} 的字幕，校验失败，触发重试。")

    return merged, missing_ids

def call_deepseek_translate(
    client: OpenAI,
    model: str,
    batch_entries: List[SubtitleEntry],
    prev_entries_orig: Optional[List[SubtitleEntry]] = None,
    prev_entries_trans: Optional[List[SubtitleEntry]] = None,
    movie_name: str = "",
    movie_bg: str = "",
    glossary: str = "",
    max_retries: int = MAX_RETRIES,
) -> tuple[List[SubtitleEntry], List[int]]: # 注意这里返回类型变了
    
    user_prompt = build_user_prompt(
        batch_entries, prev_entries_orig, prev_entries_trans, 
        movie_name, movie_bg, glossary
    )
    
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        is_last_attempt = (attempt == max_retries) # 判断是否是最后一次重试
        
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.1,  
                max_tokens=2000,  # <--- [新增] 防止幻觉导致疯狂计费
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"} if model == "deepseek-chat" else None
            )
            content = resp.choices[0].message.content or ""
            
            # 前几次重试 strict=True，最后一次重试 strict=False (接受原文兜底)
            return extract_and_validate_json(batch_entries, content, strict=not is_last_attempt)
            
        except (APITimeoutError, APIConnectionError, APIStatusError, RateLimitError, ValueError) as err:
            last_error = err
            if not is_last_attempt:
                time.sleep(RETRY_WAIT_SECONDS * attempt)
            else:
                # 只有当 API 网络彻底断开等严重错误时才抛出异常
                if not isinstance(err, ValueError): 
                    raise RuntimeError(f"网络或API接口异常，批次(序号 {batch_entries[0].index} 起)彻底失败：{last_error}")

    # 理论上不会走到这里，加个兜底
    return batch_entries, [e.index for e in batch_entries]


def decode_uploaded_file(uploaded_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return uploaded_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("文件编码无法识别，请使用 UTF-8 或 GB 编码的 SRT 文件。")


def get_nested_secret(keys: List[str]) -> Optional[str]:
    try:
        v: Any = st.secrets
        for k in keys:
            v = v[k]
        return None if v is None else str(v)
    except Exception:
        return None


def load_secrets_from_file() -> Dict[str, Any]:
    if tomllib is None:
        return {}
    try:
        secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
        if not secrets_path.exists():
            return {}
        with secrets_path.open("rb") as f:
            data = tomllib.load(f)
        return dict(data)
    except Exception:
        return {}


def get_value_from_dict(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def is_likely_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    v = value.strip()
    return v.startswith("YOUR_") or v in {"DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"}


def get_deepseek_api_key() -> Optional[str]:
    key = get_nested_secret(["DEEPSEEK_API_KEY"])
    if is_likely_placeholder(key):
        file_secrets = load_secrets_from_file()
        key = get_value_from_dict(file_secrets, ["DEEPSEEK_API_KEY"])
    return None if is_likely_placeholder(key) else str(key)


def get_feedback_settings() -> Optional[Dict[str, str]]:
    try:
        fb = st.secrets.get("feedback", {})
        if not fb or is_likely_placeholder(fb.get("smtp_host")):
            return None
        return {
            "smtp_host": fb["smtp_host"],
            "smtp_port": str(fb["smtp_port"]),
            "smtp_username": fb["smtp_username"],
            "smtp_password": str(fb["smtp_password"]),
            "from_email": fb["from_email"],
            "to_email": fb["to_email"]
        }
    except Exception:
        return None


def send_feedback_email(feedback_text: str) -> None:
    settings = get_feedback_settings()
    if not settings:
        print("未配置反馈邮箱的 SMTP 参数")
        return

    smtp_host = settings["smtp_host"]
    smtp_port = int(settings["smtp_port"])
    smtp_username = settings["smtp_username"]
    smtp_password = settings["smtp_password"]
    from_email = settings["from_email"]
    to_email = settings["to_email"]

    msg = EmailMessage()
    msg["Subject"] = "SRT 翻译工具用户反馈"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(feedback_text, charset="utf-8")

    timeout = 20
    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout) as server:
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout) as server:
                server.ehlo()
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)
    except Exception as e:
        print(f"异步发送邮件失败: {e}")


def main() -> None:
    st.set_page_config(page_title="DYY SRT 智能翻译工具", page_icon="🎬", layout="wide")
    st.title("🎬 DYY SRT 智能翻译工具 (术语协同版)")

    # 统一初始化 Session State
    if "translated_dict" not in st.session_state: st.session_state["translated_dict"] = {}
    if "current_file_name" not in st.session_state: st.session_state["current_file_name"] = ""
    if "translated_srt" not in st.session_state: st.session_state["translated_srt"] = None
    if "translated_name" not in st.session_state: st.session_state["translated_name"] = "translated.srt"
    
    # [新增 UI 状态管理] 用于术语与背景设置
    if "ui_movie_name" not in st.session_state: st.session_state["ui_movie_name"] = ""
    if "ui_movie_bg" not in st.session_state: st.session_state["ui_movie_bg"] = ""
    if "ui_glossary" not in st.session_state: st.session_state["ui_glossary"] = ""

    with st.sidebar:
        st.header("⚙️ 全局配置")
        api_key = get_deepseek_api_key()
        if not api_key:
            api_key = st.text_input("DeepSeek API Key", type="password", placeholder="sk-...")

        model = st.selectbox("模型选择", options=["deepseek-chat", "deepseek-reasoner"], index=0)
        max_workers = st.slider("并发请求数（提高速度）", min_value=1, max_value=10, value=3)
        timeout_seconds = st.slider("请求超时（秒）", min_value=30, max_value=300, value=120, step=10)
        
        st.markdown("---")
        st.header("💰 降本增效参数 (省钱核心)")
        ui_batch_size = st.slider(
            "每批最大字幕数（越大越省钱）", 
            min_value=30, max_value=120, 
            value=DEFAULT_BATCH_SIZE, # <--- 引用默认值
            step=10
        )
        ui_gap_seconds = st.slider(
            "换场停顿判定（秒，越大越省钱）", 
            min_value=3, max_value=60, 
            value=int(DEFAULT_GAP_THRESHOLD_MS / 1000), # <--- 毫秒转秒
            step=1
        )

        current_movie_name = st.session_state.get("ui_movie_name", "")
        current_movie_bg = st.session_state.get("ui_movie_bg", "")
        current_glossary = st.session_state.get("ui_glossary", "")
        
        st.markdown("---")
        st.caption(f"🔧 底层逻辑：动态切割批次，最大 `{ui_batch_size}` 条/批，遇 `{ui_gap_seconds}`秒 以上断句自动切片。")
        st.caption(f"🔄 错误重试：最大 `{MAX_RETRIES}` 次零容错校验")

        st.markdown("---")
        st.header("📩 用户反馈")
        with st.form("feedback_form", clear_on_submit=True):
            feedback_text = st.text_area("请输入你的反馈", height=120)
            submit_feedback = st.form_submit_button("提交反馈")
        if submit_feedback and feedback_text.strip():
            threading.Thread(target=send_feedback_email, args=(feedback_text.strip(),)).start()
            st.success("反馈已发送！")

    st.markdown("### 第一步：上传剧本文件")
    uploaded_file = st.file_uploader("上传 .srt 文件", type=["srt"], label_visibility="collapsed")
    file_bytes = None
    entries = []
    
    if uploaded_file:
        file_bytes = uploaded_file.read() 
        try:
            content = decode_uploaded_file(file_bytes) 
            entries = parse_srt(content)
        except Exception as err:
            st.error(f"SRT 解析失败：{err}")
            st.stop()

        # --- [UI 阶段二] 背景设定与术语工作流 ---
        st.markdown("---")
        st.markdown("### 第二步：影视设定与术语一致性 (AI 辅助)")
        st.info("在正式翻译前锁定专有名词，可极大避免角色“乱改名”的现象。")
        
        col1, col2 = st.columns(2)
        with col1:
            movie_name_input = st.text_input("影视名称 (选填)", value=st.session_state["ui_movie_name"], placeholder="例如：奥本海默")
        with col2:
            movie_bg_input = st.text_input("背景/风格 (选填)", value=st.session_state["ui_movie_bg"], placeholder="例如：二战、严肃传记风格")
            
        glossary_input = st.text_area("📝 专有名词对照表", value=st.session_state["ui_glossary"], placeholder="例如：\nJohn: 约翰\nFBI: 联邦调查局", height=120)
        
        c1, c2, _ = st.columns([2, 1, 3])
        with c1:
            if st.button("✨ AI 阅读前150句自动提取核心术语", type="secondary", use_container_width=True):
                if not api_key:
                    st.error("请先在左侧配置 API Key！")
                else:
                    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1", timeout=timeout_seconds)
                    with st.spinner("AI 正在光速阅片提取..."):
                        extracted_glossary = auto_extract_glossary(client, model, entries)
                        if "失败" not in extracted_glossary:
                            st.session_state["ui_glossary"] = extracted_glossary
                            st.rerun() # 触发重新渲染，刷新输入框
                        else:
                            st.error(extracted_glossary)
                            
        # 同步最新的 UI 输入到 Session
        st.session_state["ui_movie_name"] = movie_name_input
        st.session_state["ui_movie_bg"] = movie_bg_input
        st.session_state["ui_glossary"] = glossary_input

        # 获取缓存路径并对比校验
        current_cache_file = get_cache_file_path(
            file_bytes, model, movie_bg_input, glossary_input, 
            ui_batch_size, ui_gap_seconds * 1000
        )
        
        with c2:
            if st.button("🗑️ 清除当前缓存", use_container_width=True):
                if current_cache_file.exists(): current_cache_file.unlink()
                st.session_state["translated_dict"] = {}
                st.rerun()

        if uploaded_file.name != st.session_state["current_file_name"] or "cache_file" not in st.session_state or st.session_state["cache_file"] != current_cache_file:
            st.session_state["current_file_name"] = uploaded_file.name
            st.session_state["cache_file"] = current_cache_file
            
            cached_progress = load_progress_from_local(current_cache_file)
            st.session_state["translated_dict"] = cached_progress
            st.session_state["translated_srt"] = None
            if cached_progress:
                st.toast(f"检测到本地缓存！已自动为您恢复 {len(cached_progress)} 个批次的进度。")

        # --- [UI 阶段四] 执行翻译 ---
        st.markdown("---")
        st.markdown("### 第三步：正式开始并发翻译")
        start_translate = st.button("🚀 开始翻译 / 继续翻译", type="primary", use_container_width=True)

        if start_translate:
            if not api_key:
                st.error("请先在侧边栏填写 DeepSeek API Key。")
                st.stop()

            client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1",
                timeout=timeout_seconds,
            )

            # [应用阶段四] 替换为智能场景切片算法
            batches = smart_chunk_entries(
                entries, 
                max_batch_size=ui_batch_size, 
                gap_threshold_ms=ui_gap_seconds * 1000  # 秒转毫秒
            )
            total_batches = len(batches)
            start_time = time.time()
            
            pending_batches = []
            for i, batch in enumerate(batches):
                if i not in st.session_state["translated_dict"]:
                    # [滑动窗口优化] 带入前一幕的情境
                    prev_orig = batches[i-1][-3:] if i > 0 else None
                    prev_trans = st.session_state["translated_dict"].get(i-1, [])[-3:] if i > 0 else None
                    
                    if prev_orig and prev_trans and len(prev_orig) != len(prev_trans):
                        prev_trans = None
                        
                    pending_batches.append((i, batch, prev_orig, prev_trans))

            completed_batches_count = total_batches - len(pending_batches)

            st.info(f"💡 智能切片分析：共 {len(entries)} 条字幕，按语境停顿划分为 {total_batches} 幕。已完成 {completed_batches_count} 幕，待处理 {len(pending_batches)} 幕。")
            progress = st.progress(completed_batches_count / total_batches if total_batches > 0 else 0.0, text="准备处理...")
            status = st.empty()

            if not pending_batches:
                status.success("所有内容均已翻译完毕并在缓存中，直接生成文件。")
            else:
                status.info(f"正在以并发量 {max_workers} 翻译中，进度实时持久化保护...")
                failed = False

                if "missing_ids" not in st.session_state:
                    st.session_state["missing_ids"] = []
                
                def process_batch(idx, batch, p_orig, p_trans):
                    # 现在返回值包含了 missing_ids
                    translated_batch, missing = call_deepseek_translate(
                        client, model, batch, 
                        prev_entries_orig=p_orig, 
                        prev_entries_trans=p_trans, 
                        movie_name=current_movie_name, 
                        movie_bg=current_movie_bg, 
                        glossary=current_glossary, 
                        max_retries=MAX_RETRIES
                    )
                    return idx, translated_batch, missing

                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(process_batch, i, batch, orig, trans): i for i, batch, orig, trans in pending_batches}
            
                    for future in concurrent.futures.as_completed(futures):
                        try: 
                            # 接收三个返回值
                            idx, translated_batch, missing_ids = future.result()
                            
                            st.session_state["translated_dict"][idx] = translated_batch
                            save_progress_to_local(st.session_state["cache_file"], st.session_state["translated_dict"])
                            
                            # 如果有漏翻，记录到 session 里面
                            if missing_ids:
                                st.session_state["missing_ids"].extend(missing_ids)

                            completed_batches_count += 1
                            progress_val = min(completed_batches_count / total_batches, 1.0)
                            
                            elapsed_time = time.time() - start_time
                            batches_processed_this_run = completed_batches_count - (total_batches - len(pending_batches))
                            if batches_processed_this_run > 0:
                                avg_time_per_batch = elapsed_time / batches_processed_this_run
                                remaining_batches = total_batches - completed_batches_count
                                eta_seconds = int(avg_time_per_batch * remaining_batches)
                                eta_str = f"约 {eta_seconds // 60}分 {eta_seconds % 60}秒"
                            else:
                                eta_str = "计算中..."

                            progress.progress(
                                progress_val, 
                                text=f"翻译进度：{completed_batches_count}/{total_batches} 幕 ⏳ 预计剩余时间：{eta_str}"
                            )
                        except Exception as err:
                            failed = True
                            status.error(f"⚠️ 发生严重错误（网络或API崩溃），已停止。错误详情：{err}")
                            executor.shutdown(wait=False, cancel_futures=True) 
                            break

                if failed:
                    st.warning("翻译已中止。您可以稍作等待后，再次点击【继续翻译】按钮，程序将自动从断点继续。")
                    st.stop()

            translated_all = []
            for i in range(total_batches):
                translated_all.extend(st.session_state["translated_dict"][i])

            output_srt = entries_to_srt(translated_all)
            original_name = uploaded_file.name.rsplit(".", 1)[0]
            output_name = f"{original_name}_zh.srt"

            st.session_state["translated_srt"] = output_srt
            st.session_state["translated_name"] = output_name

            status.success("🎉 全部翻译完成！")

            # 【新增：漏翻提示 UI】
            missing_list = st.session_state.get("missing_ids", [])
            if missing_list:
                st.warning(f"⚠️ 提示：AI 在翻译过程中漏翻了 {len(missing_list)} 条极其简短或特殊的字幕。为了保证时间轴不乱，系统已保留它们的英文原文。")
                st.info(f"请在下载后的 SRT 文件中，搜索并手动补全以下字幕序号：\n {', '.join(map(str, sorted(missing_list)))}")
            else:
                st.info("完美！本次翻译没有检测到任何 AI 漏翻的字幕。")
                
            # 清理 missing_ids 以备下次翻译
            st.session_state["missing_ids"] = []

    if st.session_state["translated_srt"]:
        with st.expander("预览翻译结果", expanded=False):
            st.text(st.session_state["translated_srt"][:1500] + "\n\n...（省略后续内容）")
            
        st.download_button(
            label="下载最终翻译版 SRT ⏬",
            data=st.session_state["translated_srt"].encode("utf-8"),
            file_name=st.session_state["translated_name"],
            mime="text/plain",
            use_container_width=True,
        )

if __name__ == "__main__":
    main()