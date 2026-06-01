"""
DYY SRT 智能翻译工具 — Streamlit UI entry point.

All non-UI logic lives in the ``src/`` package:

.. rubric:: ``src/core/``
   - ``constants.py``  — shared config constants
   - ``models.py``     — ``SubtitleEntry`` dataclass
   - ``text_processor.py``  — Chinese detection, smart wrapping, ASS tag mask
   - ``chunker.py``   — scene-aware batch splitting

.. rubric:: ``src/parsers/``
   - ``srt_parser.py`` — SRT file parse + serialise
   - ``ass_parser.py`` — ASS/SSA file parse + serialise

.. rubric:: ``src/translator/``
   - ``prompts.py``   — system / user prompt builders
   - ``engine.py``    — API call, validation, retry, isolation fallback
   - ``cache.py``     — MD5-based local cache
   - ``glossary.py``  — auto-extract terminology from first N entries
"""

import concurrent.futures
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from openai import OpenAI

# ── Core domain ─────────────────────────────────────────────────────────────────
from src.core.models import SubtitleEntry
from src.core.text_processor import contains_chinese
from src.core.chunker import smart_chunk_entries, compress_full_script
from src.core.constants import DEFAULT_BATCH_SIZE, DEFAULT_GAP_THRESHOLD_MS, MAX_RETRIES

# ── Parsers ────────────────────────────────────────────────────────────────────
from src.parsers.srt_parser import parse_srt, entries_to_srt, decode_uploaded_file
from src.parsers.ass_parser import parse_ass, entries_to_ass

# ── Translator engine ──────────────────────────────────────────────────────────
from src.translator.engine import call_deepseek_translate, _toast
from src.translator.cache import (
    get_cache_file_path,
    save_progress_to_local,
    load_progress_from_local,
)
from src.translator.glossary import auto_extract_glossary

# ── Secret / config helpers ────────────────────────────────────────────────────

try:
    import tomllib
except ImportError:
    tomllib = None  # type: ignore


def get_nested_secret(keys: List[str]) -> Optional[str]:
    try:
        v: Any = st.secrets
        for k in keys:
            v = v[k]
        return None if v is None else str(v)
    except Exception:
        return None


def is_likely_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True
    v = value.strip()
    return v.startswith("YOUR_") or v in {"DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_API_KEY"}


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
            "to_email": fb["to_email"],
        }
    except Exception:
        return None


def send_feedback_email(feedback_text: str) -> None:
    import smtplib
    from email.message import EmailMessage

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


# ── Runtime config (can be overridden via secrets.toml) ───────────────────────

API_BASE_URL = st.secrets.get("DEEPSEEK_API_BASE_URL", "https://api.deepseek.com")
AVAILABLE_MODELS = st.secrets.get(
    "AVAILABLE_MODELS", ["deepseek-v4-flash", "deepseek-v4-pro"]
)
JSON_SUPPORTED_MODELS = st.secrets.get("JSON_SUPPORTED_MODELS", ["deepseek-v4-flash"])


# ── Main UI ────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="DYY SRT 智能翻译工具", page_icon="🎬", layout="wide"
    )
    st.title("🎬 DYY SRT 智能翻译工具 (术语协同版)")

    # ── Session state defaults ─────────────────────────────────────────────────
    for key, default in [
        ("translated_dict", {}),
        ("current_file_name", ""),
        ("translated_srt", None),
        ("translated_name", "translated.srt"),
        ("ui_movie_name", ""),
        ("ui_movie_bg", ""),
        ("ui_glossary", ""),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar: global configuration ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 全局配置")
        api_key = get_deepseek_api_key()
        if not api_key:
            api_key = st.text_input(
                "DeepSeek API Key", type="password", placeholder="sk-..."
            )

        model = st.selectbox(
            "模型选择", options=AVAILABLE_MODELS, index=0
        )
        max_workers = st.slider(
            "并发请求数（提高速度）", min_value=1, max_value=10, value=6
        )
        timeout_seconds = st.slider(
            "请求超时（秒）", min_value=30, max_value=300, value=120, step=10
        )

        st.markdown("---")
        st.header("🧠 模型行为控制")
        ui_use_thinking = st.toggle(
            "🤔 开启思考模式 (Thinking Mode)",
            value=False,
            help="让模型在输出前先推演思维链。大幅提升复杂双关语的翻译水平，"
            "但会显著降低速度并增加少量计费。普通电影建议关闭。",
        )

        if ui_use_thinking:
            ui_reasoning_effort = st.selectbox(
                "思考强度",
                options=["high", "max"],
                index=0,
                help="high: 默认思考强度；max: 最高思考强度（针对极其烧脑的悬疑剧）。",
            )
            ui_temperature = 0.1
            st.info(
                "ℹ️ 已开启思考模式，温度(Temperature)已被模型底层接管。"
            )
        else:
            ui_reasoning_effort = "high"
            ui_temperature = st.slider(
                "🌡️ 温度 (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=0.1,
                step=0.1,
                help="值越低，翻译越严谨一致；值越高，用词越丰富有创造力"
                "（影视字幕建议 0.1 - 0.3）。",
            )

        st.markdown("---")
        st.header("💰 降本增效参数 (省钱核心)")
        ui_god_mode = st.toggle(
            "👁️ 开启上帝视角 (全量剧本上下文注入)",
            value=True,
            help="极大幅提升翻译连贯性与伏笔处理。由于DeepSeek具有上下文缓存，"
            "开启此项对整体成本影响极小（强烈推荐开启）。",
        )
        ui_skip_chinese = st.toggle(
            "⏭️ 跳过已包含中文的字幕",
            value=True,
            help="如果上传的字幕中已经包含中文，将自动跳过这些条目，"
            "只翻译纯外文部分，节省时间和费用。",
        )
        ui_batch_size = st.slider(
            "每批最大字幕数（越大越省钱）",
            min_value=30,
            max_value=120,
            value=DEFAULT_BATCH_SIZE,
            step=10,
        )
        ui_gap_seconds = st.slider(
            "换场停顿判定（秒，越大越省钱）",
            min_value=3,
            max_value=60,
            value=int(DEFAULT_GAP_THRESHOLD_MS / 1000),
            step=1,
        )

        current_movie_name = st.session_state.get("ui_movie_name", "")
        current_movie_bg   = st.session_state.get("ui_movie_bg", "")
        current_glossary  = st.session_state.get("ui_glossary", "")

        st.markdown("---")
        st.caption(
            f"🔧 底层逻辑：动态切割批次，最大 `{ui_batch_size}` 条/批，"
            f"遇 `{ui_gap_seconds}` 秒以上断句自动切片。"
        )
        st.caption(f"🔄 错误重试：最大 `{MAX_RETRIES}` 次零容错校验")

        st.markdown("---")
        st.header("📩 用户反馈")
        with st.form("feedback_form", clear_on_submit=True):
            feedback_text = st.text_area("请输入你的反馈", height=120)
            submit_feedback = st.form_submit_button("提交反馈")
        if submit_feedback and feedback_text.strip():
            threading.Thread(
                target=send_feedback_email, args=(feedback_text.strip(),)
            ).start()
            st.success("反馈已发送！")

    # ── Step 1: upload ─────────────────────────────────────────────────────────
    st.markdown("### 第一步：上传剧本文件")
    uploaded_file = st.file_uploader(
        "上传字幕文件", type=["srt", "ass"], label_visibility="collapsed"
    )
    file_bytes: Optional[bytes] = None
    entries: List[SubtitleEntry] = []

    if uploaded_file:
        file_bytes = uploaded_file.read()
        try:
            content = decode_uploaded_file(file_bytes)
            file_ext = uploaded_file.name.lower().split(".")[-1]
            st.session_state["file_ext"] = file_ext

            if file_ext == "ass":
                entries, ass_header = parse_ass(content)
                st.session_state["ass_header"] = ass_header
            else:
                entries = parse_srt(content)

            full_script_text = compress_full_script(entries)

            if ui_god_mode and len(entries) >= 3000:
                st.toast(
                    "⚠️ 剧本极长！由于上帝视角需传输全剧上下文，"
                    "若稍后报错请关闭此功能。",
                    icon="🔥",
                )
        except Exception as err:
            st.error(f"SRT 解析失败：{err}")
            st.stop()

        # ── Step 2: movie context + glossary ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 第二步：影视设定与术语一致性 (AI 辅助)")
        st.info("在正式翻译前锁定专有名词，可极大避免角色「乱改名」的现象。")

        col1, col2 = st.columns(2)
        with col1:
            movie_name_input = st.text_input(
                "影视名称 (选填)",
                value=st.session_state["ui_movie_name"],
                placeholder="例如：奥本海默",
            )
        with col2:
            movie_bg_input = st.text_input(
                "背景/风格 (选填)",
                value=st.session_state["ui_movie_bg"],
                placeholder="例如：二战、严肃传记风格",
            )

        glossary_input = st.text_area(
            "📝 专有名词对照表",
            value=st.session_state["ui_glossary"],
            placeholder="例如：\nJohn: 约翰\nFBI: 联邦调查局",
            height=120,
        )

        c1, c2, _ = st.columns([2, 1, 3])
        with c1:
            if st.button(
                "✨ AI 阅读前150句自动提取核心术语",
                type="secondary",
                use_container_width=True,
            ):
                if not api_key:
                    st.error("请先在左侧配置 API Key！")
                else:
                    client = OpenAI(
                        api_key=api_key,
                        base_url=API_BASE_URL,
                        timeout=timeout_seconds,
                    )
                    with st.spinner("AI 正在光速阅片提取..."):
                        extracted_glossary = auto_extract_glossary(
                            client, model, entries
                        )
                        if "失败" not in extracted_glossary:
                            st.session_state["ui_glossary"] = extracted_glossary
                            st.rerun()
                        else:
                            st.error(extracted_glossary)

        # Sync UI state
        st.session_state["ui_movie_name"] = movie_name_input
        st.session_state["ui_movie_bg"]   = movie_bg_input
        st.session_state["ui_glossary"]  = glossary_input

        # ── Cache management ────────────────────────────────────────────────────
        current_cache_file = get_cache_file_path(
            file_bytes, model, movie_bg_input, glossary_input,
            ui_batch_size, ui_gap_seconds * 1000, ui_god_mode,
            ui_temperature, ui_use_thinking, ui_reasoning_effort,
            ui_skip_chinese,
        )

        with c2:
            if st.button("🗑️ 清除当前缓存", use_container_width=True):
                if current_cache_file.exists():
                    current_cache_file.unlink()
                st.session_state["translated_dict"] = {}
                st.rerun()

        if (
            uploaded_file.name != st.session_state["current_file_name"]
            or "cache_file" not in st.session_state
            or st.session_state["cache_file"] != current_cache_file
        ):
            st.session_state["current_file_name"] = uploaded_file.name
            st.session_state["cache_file"] = current_cache_file

            cached_progress = load_progress_from_local(current_cache_file)
            st.session_state["translated_dict"] = cached_progress
            st.session_state["translated_srt"] = None
            if cached_progress:
                st.toast(
                    f"检测到本地缓存！"
                    f"已自动为您恢复 {len(cached_progress)} 个批次的进度。"
                )

        # ── Step 3: run translation ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 第三步：正式开始并发翻译")
        start_translate = st.button(
            "🚀 开始翻译 / 继续翻译",
            type="primary",
            use_container_width=True,
        )

        if start_translate:
            if not api_key:
                st.error("请先在侧边栏填写 DeepSeek API Key。")
                st.stop()

            client = OpenAI(
                api_key=api_key,
                base_url=API_BASE_URL,
                timeout=timeout_seconds,
            )

            batches = smart_chunk_entries(
                entries,
                max_batch_size=ui_batch_size,
                gap_threshold_ms=ui_gap_seconds * 1000,
            )
            total_batches = len(batches)
            MAX_AUTO_PATCH_ROUNDS = 2

            status = st.empty()
            progress = st.progress(0.0, text="准备处理...")

            # ── Self-healing outer loop ─────────────────────────────────────────
            for current_round in range(MAX_AUTO_PATCH_ROUNDS + 1):
                pending_tasks: List[Dict] = []

                # Build task queue from cache
                for i, batch in enumerate(batches):
                    prev_orig = batches[i - 1][-40:] if i > 0 else None
                    prev_trans = (
                        st.session_state["translated_dict"].get(i - 1, [])[-40:]
                        if i > 0
                        else None
                    )
                    if prev_orig and prev_trans and len(prev_orig) != len(prev_trans):
                        prev_trans = None

                    if i not in st.session_state["translated_dict"]:
                        # New batch
                        if ui_skip_chinese:
                            needs_translation: List[SubtitleEntry] = []
                            pre_translated: List[SubtitleEntry] = []
                            for e in batch:
                                if contains_chinese(e.text):
                                    pre_translated.append(e)
                                else:
                                    needs_translation.append(e)

                            if not needs_translation:
                                st.session_state["translated_dict"][i] = pre_translated
                                continue

                            pending_tasks.append({
                                "idx": i,
                                "entries": needs_translation,
                                "pre_translated": pre_translated,
                                "p_orig": prev_orig,
                                "p_trans": prev_trans,
                                "is_patch": False,
                            })
                        else:
                            pending_tasks.append({
                                "idx": i,
                                "entries": batch,
                                "pre_translated": [],
                                "p_orig": prev_orig,
                                "p_trans": prev_trans,
                                "is_patch": False,
                            })
                    else:
                        # Check for fallbacks in cached batch
                        cached_batch = st.session_state["translated_dict"][i]
                        orig_dict = {e.index: e for e in batch}
                        missing_entries = [
                            orig_dict[ce.index]
                            for ce in cached_batch
                            if getattr(ce, "is_fallback", False)
                        ]

                        if missing_entries:
                            task_temp = (
                                ui_temperature
                                if current_round == 0
                                else min(ui_temperature + 0.2, 1.0)
                            )
                            pending_tasks.append({
                                "idx": i,
                                "entries": missing_entries,
                                "p_orig": prev_orig,
                                "p_trans": prev_trans,
                                "is_patch": True,
                                "patch_temp": task_temp,
                            })

                if not pending_tasks:
                    if current_round == 0:
                        # Defensive guard: verify the cache actually covers all entries
                        cached_count = sum(
                            len(st.session_state["translated_dict"].get(i, []))
                            for i in range(total_batches)
                        )
                        if cached_count != len(entries):
                            status.warning(
                                f"检测到不完整的缓存（{cached_count}/{len(entries)} 条），"
                                "强制执行一轮翻译补全..."
                            )
                            for i, batch in enumerate(batches):
                                pending_tasks.append({
                                    "idx": i,
                                    "entries": batch,
                                    "pre_translated": [],
                                    "p_orig": None,
                                    "p_trans": None,
                                    "is_patch": False,
                                })
                        else:
                            status.success("所有内容均已存在完整缓存，无需重复请求。")
                            break
                    else:
                        status.success(
                            f"自动补漏大满贯！"
                            f"所有遗漏已在第 {current_round} 轮被彻底修复。"
                        )
                        break
                else:
                    # pending_tasks is non-empty — proceed to translation (no break)
                    pass

                if current_round == 0:
                    status.info(
                        f"🚀 第一轮主力翻译中 (并发量 {max_workers})... "
                        "遇到漏翻将自动启动二阶段保护。"
                    )
                    completed_batches = total_batches - len(pending_tasks)
                    total_tasks_this_round = total_batches
                else:
                    status.warning(
                        f"🔍 触发第 {current_round} 轮精准自愈补漏："
                        f"自动狙击 {sum(len(t['entries']) for t in pending_tasks)} 条顽固字幕..."
                    )
                    completed_batches = 0
                    total_tasks_this_round = len(pending_tasks)
                    max_workers = max(1, max_workers // 2)

                failed = False

                def process_batch(task: Dict) -> tuple:
                    temp = task.get("patch_temp", ui_temperature)
                    translated_segment, _ = call_deepseek_translate(
                        client,
                        model,
                        task["entries"],
                        prev_entries_orig=task["p_orig"],
                        prev_entries_trans=task["p_trans"],
                        movie_name=current_movie_name,
                        movie_bg=current_movie_bg,
                        glossary=current_glossary,
                        max_retries=MAX_RETRIES,
                        full_script_text=full_script_text,
                        use_god_mode=ui_god_mode,
                        temperature=temp,
                        use_thinking=ui_use_thinking,
                        reasoning_effort=ui_reasoning_effort,
                    )
                    return task, translated_segment

                # ── Execute with ThreadPoolExecutor ──────────────────────────────
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers
                ) as executor:
                    futures = {
                        executor.submit(process_batch, task): task
                        for task in pending_tasks
                    }

                    pending_count = 0

                    for future in concurrent.futures.as_completed(futures):
                        try:
                            task, translated_segment = future.result()
                            idx = task["idx"]

                            # Detect isolation fallback and notify in the main thread
                            # (calling _toast from a worker would trigger ScriptRunContext warnings)
                            if any(getattr(e, "_isolation_fallback", False) for e in translated_segment):
                                _toast(
                                    f"检测到 ID {task['entries'][0].index} 附近存在顽固合并倾向，"
                                    "已启动【物理隔离模式】强行对齐时间轴..."
                                )

                            if task["is_patch"]:
                                cached_batch = st.session_state["translated_dict"][idx]
                                segment_dict = {e.index: e for e in translated_segment}
                                for ci, cached_entry in enumerate(cached_batch):
                                    if cached_entry.index in segment_dict:
                                        cached_batch[ci] = segment_dict[cached_entry.index]
                                st.session_state["translated_dict"][idx] = cached_batch
                            else:
                                if task.get("pre_translated"):
                                    combined = translated_segment + task["pre_translated"]
                                    combined.sort(key=lambda x: x.index)
                                    st.session_state["translated_dict"][idx] = combined
                                else:
                                    st.session_state["translated_dict"][idx] = translated_segment

                            pending_count += 1
                            if pending_count >= 5:
                                save_progress_to_local(
                                    st.session_state["cache_file"],
                                    st.session_state["translated_dict"],
                                )
                                pending_count = 0

                            completed_batches += 1
                            progress_val = min(
                                completed_batches / max(1, total_tasks_this_round),
                                1.0,
                            )
                            label = (
                                "自愈补漏"
                                if current_round > 0
                                else "主线进度"
                            )
                            progress.progress(
                                progress_val,
                                text=f"【{label}】 {completed_batches}/{total_tasks_this_round} 任务已完成...",
                            )
                        except Exception as err:
                            failed = True
                            status.error(
                                f"⚠️ 发生严重错误（网络或API崩溃），已停止。错误详情：{err}"
                            )
                            if pending_count > 0:
                                save_progress_to_local(
                                    st.session_state["cache_file"],
                                    st.session_state["translated_dict"],
                                )
                            executor.shutdown(wait=False, cancel_futures=True)
                            break

                    if pending_count > 0:
                        save_progress_to_local(
                            st.session_state["cache_file"],
                            st.session_state["translated_dict"],
                        )

                if failed:
                    st.warning(
                        "翻译已中止。您可以稍作等待后，再次点击【开始翻译/继续翻译】"
                        "按钮，程序将自动从断点继续。"
                    )
                    st.stop()

            # ── Finalise results ─────────────────────────────────────────────────
            final_missing_ids: List[int] = []
            for i in range(total_batches):
                for entry in st.session_state["translated_dict"].get(i, []):
                    if getattr(entry, "is_fallback", False):
                        final_missing_ids.append(entry.index)
            st.session_state["missing_ids"] = final_missing_ids

            translated_all: List[SubtitleEntry] = []
            for i in range(total_batches):
                translated_all.extend(st.session_state["translated_dict"][i])

            original_name = uploaded_file.name.rsplit(".", 1)[0]

            if st.session_state.get("file_ext") == "ass":
                output_str = entries_to_ass(
                    translated_all, st.session_state.get("ass_header", "")
                )
                output_name = f"{original_name}_zh.ass"
            else:
                output_str = entries_to_srt(translated_all)
                output_name = f"{original_name}_zh.srt"

            st.session_state["translated_srt"] = output_str
            st.session_state["translated_name"] = output_name
            status.success("🎉 全部翻译完成！")

            missing_list = st.session_state.get("missing_ids", [])
            if missing_list:
                st.warning(
                    f"⚠️ 提示: AI 在翻译过程中漏翻了 {len(missing_list)} 条极其简短或特殊的字幕。"
                    "为了保证时间轴不乱，系统已保留它们的英文原文。"
                )
                st.info(
                    f"请在下载后的 SRT 文件中，搜索并手动补全以下字幕序号：\n"
                    f"{', '.join(map(str, sorted(missing_list)))}"
                )
            else:
                st.info("完美！本次翻译没有检测到任何 AI 漏翻的字幕。")

            st.session_state["missing_ids"] = []

    # ── Download section (always visible when results exist) ───────────────────
    if st.session_state["translated_srt"]:
        with st.expander("预览翻译结果", expanded=False):
            preview = st.session_state["translated_srt"]
            st.text(preview[:1500] + "\n\n...（省略后续内容）")

        st.download_button(
            label="下载最终翻译版 SRT ⏬",
            data=st.session_state["translated_srt"].encode("utf-8"),
            file_name=st.session_state["translated_name"],
            mime="text/plain",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
