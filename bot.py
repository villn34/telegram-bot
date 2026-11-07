#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
بات تلگرام شخصی — دستیار آموزشی با چند مدل هوش مصنوعی
مدل‌ها از سه جا می‌آیند: Tooken Club (پروایدر اصلی — با ۵ میلیون توکن کاربر)،
OpenRouter (رایگان :free + پولی) و Gemini مستقیم گوگل.

مدل‌های Tooken (از tooken.club/v1 — همه از موجودی Tooken کم می‌شود):
    - tooken-deepseek       : deepseek-v4.1-flash-1017 — تأییدشده در tooken.club/models — مدل پیش‌فرض ⭐
    - tooken-deepseek-pro   : deepseek-v4-pro — شناسه از روی نام کاتالوگ
    - tooken-gpt-luna       : gpt-6-luna — تأییدشده (نسخه 5.6 هم هنوز کار می‌کند)
    - tooken-gpt-sol        : gpt-6.1-sol — شناسه از روی نام کاتالوگ
    - tooken-claude-sonnet  : claude-sonnet-5-5 — تأییدشده
    - tooken-claude-haiku   : claude-haiku-5-5 — شناسه از روی نام کاتالوگ
    - tooken-kimi           : kimi-k3.1-pro — تأییدشده
    - tooken-gemini         : gemini-3.8-flash — شناسه از روی نام کاتالوگ
  توجه: در کاتالوگ Tooken مدلی به نام Qwen نیست (به‌جایش GLM و Kimi هست).
  اگر مدلی خطای 404 داد، شناسه دقیقش را در tooken.club/models چک کنید
  و فقط همان یک خط model_id را در MODELS عوض کنید.

مدل‌ها:
  رایگان (OpenRouter :free):
    - deepseek-free : deepseek/deepseek-v4.1-flash-1017:free
    - qwen-free     : qwen/qwen3-32b:free
    - gemma-free    : google/gemma-3-27b-it:free
  رایگان (Gemini مستقیم):
    - gemini-flash  : gemini-3.6-flash  (حدود ۲۰ درخواست در روز)
    - gemini-lite   : gemini-3.5-flash-lite (حدود ۵۰۰ درخواست در روز) — مدل پیش‌فرض جایگزین
  پولی (OpenRouter — فقط با شارژ اعتبار کار می‌کند):
    - gpt-luna      : openai/gpt-5.6-luna
    - claude-sonnet : anthropic/claude-sonnet-5

نکته: NotebookLM API ندارد و نمی‌تواند در بات بیاید.

اجرا:
  pip install -r requirements.txt
  cp .env.example .env   # و کلیدها را پر کنید
  python bot.py

کلیدها را هرگز در کد ننویسید — فقط از متغیرهای محیطی (.env) خوانده می‌شوند.
"""

import json
import logging
import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
STATE_FILE = BASE_DIR / "user_state.json"

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
# کلید Tooken Club — از داشبورد tooken.club بگیرید.
# هرگز کلید واقعی را در کد یا در چت نفرستید؛ فقط در .env بگذارید.
TOOKEN_API_KEY = os.getenv("TOOKEN_API_KEY", "").strip()
TOOKEN_BASE_URL = "https://tooken.club/v1"

# مثال: ALLOWED_USER_IDS=123456789,987654321
_allowed_raw = os.getenv("ALLOWED_USER_IDS", "").strip()
ALLOWED_USER_IDS = set()
if _allowed_raw:
    for part in _allowed_raw.split(","):
        part = part.strip()
        if part.isdigit():
            ALLOWED_USER_IDS.add(int(part))

DEFAULT_MODEL = "tooken-deepseek"  # DeepSeek از طریق Tooken — قوی و ارزان، از ۵ میلیون توکن کاربر
FALLBACK_MODEL = "gemini-lite"
MAX_HISTORY = 10  # فقط ۱۰ پیام آخر برای حافظه مکالمه نگه داشته می‌شود
MAX_MESSAGE_LEN = 4096

SYSTEM_PROMPT = (
    "تو یک معلم و دستیار آموزشی صبور و دقیق هستی. "
    "به زبان فارسی ساده و روان توضیح بده، مگر اینکه کاربر به انگلیسی بنویسد — "
    "در آن صورت به انگلیسی جواب بده. "
    "مفاهیم را قدم‌به‌قدم و با مثال‌های ملموس توضیح بده، "
    "در پایان هر جواب یک مثال یا یک سوال تمرینی بده، "
    "و اگر مطمئن نیستی، صریح بگو که مطمئن نیستی. "
    "جواب‌ها را کوتاه و قابل استفاده برای تولید محتوای آموزشی نگه دار."
)

# ---------------------------------------------------------------------------
# مدل‌ها
# provider = "openrouter" یا "gemini" یا "tooken"
# ---------------------------------------------------------------------------
MODELS = {
    "deepseek-free": {
        "label": "DeepSeek V4.1 Flash — رایگان",
        "provider": "openrouter",
        "model_id": "deepseek/deepseek-v4.1-flash-1017:free",
        "free": True,
        "desc": "قوی‌ترین رایگان — استدلال، ریاضی و کد. مدل پیش‌فرض.",
    },
    "qwen-free": {
        "label": "Qwen 3 32B — رایگان",
        "provider": "openrouter",
        "model_id": "qwen/qwen3-32b:free",
        "free": True,
        "desc": "رایگان — خوب برای متن عمومی و چندزبانه.",
    },
    "gemma-free": {
        "label": "Gemma 3 27B — رایگان",
        "provider": "openrouter",
        "model_id": "google/gemma-3-27b-it:free",
        "free": True,
        "desc": "رایگان — مدل باز گوگل، سبک و سریع.",
    },
    "gemini-flash": {
        "label": "Gemini 3.6 Flash — رایگان (مستقیم)",
        "provider": "gemini",
        "model_id": "gemini-3.6-flash",
        "free": True,
        "desc": "رایگان مستقیم از گوگل — حدود ۲۰ درخواست در روز.",
    },
    "gemini-lite": {
        "label": "Gemini 3.5 Flash-Lite — رایگان (مستقیم)",
        "provider": "gemini",
        "model_id": "gemini-3.5-flash-lite",
        "free": True,
        "desc": "رایگان مستقیم از گوگل — حدود ۵۰۰ درخواست در روز. مدل جایگزین پیش‌فرض.",
    },
    "tooken-deepseek": {
        "label": "DeepSeek V4.1 Flash — Tooken ⭐",
        "provider": "tooken",
        "model_id": "deepseek-v4.1-flash-1017",
        "free": False,
        "desc": "مدل پیش‌فرض — قوی و ارزان، از ۵ میلیون توکن Tooken شما کم می‌شود. شناسه تأییدشده در tooken.club/models.",
    },
    "tooken-deepseek-pro": {
        "label": "DeepSeek V4 Pro — Tooken",
        "provider": "tooken",
        "model_id": "deepseek-v4-pro",
        "free": False,
        "desc": "پرچمدار DeepSeek برای استدلال و کد. شناسه از روی نام کاتالوگ است؛ اگر 404 داد در داشبورد چک کنید.",
    },
    "tooken-gpt-luna": {
        "label": "GPT-6 Luna — Tooken",
        "provider": "tooken",
        "model_id": "gpt-6-luna",
        "free": False,
        "desc": "جدیدترین Luna اوپن‌ای‌آی — تأییدشده در tooken.club/models. نسخه 5.6 هم هنوز کار می‌کند.",
    },
    "tooken-gpt-sol": {
        "label": "GPT-6.1 Sol — Tooken",
        "provider": "tooken",
        "model_id": "gpt-6.1-sol",
        "free": False,
        "desc": "قوی‌ترین GPT در کاتالوگ Tooken — متن، کد و استدلال. شناسه از روی نام کاتالوگ است.",
    },
    "tooken-claude-sonnet": {
        "label": "Claude Sonnet 5.5 — Tooken",
        "provider": "tooken",
        "model_id": "claude-sonnet-5-5",
        "free": False,
        "desc": "بهترین برای ویرایش متن و کد — تأییدشده در tooken.club/models.",
    },
    "tooken-claude-haiku": {
        "label": "Claude Haiku 5.5 — Tooken",
        "provider": "tooken",
        "model_id": "claude-haiku-5-5",
        "free": False,
        "desc": "سریع و سبک از خانواده Claude. شناسه از روی نام کاتالوگ است.",
    },
    "tooken-kimi": {
        "label": "Kimi K3.1 Pro — Tooken",
        "provider": "tooken",
        "model_id": "kimi-k3.1-pro",
        "free": False,
        "desc": "مدل Moonshot — تأییدشده در tooken.club/models. همان مدلی که دموی پارچه را ساخت.",
    },
    "tooken-gemini": {
        "label": "Gemini 3.8 Flash — Tooken",
        "provider": "tooken",
        "model_id": "gemini-3.8-flash",
        "free": False,
        "desc": "از طریق Tooken (پولی) — نسخه رایگان مستقیم گوگل هم با gemini-flash هست. شناسه از روی نام کاتالوگ است.",
    },
    "gpt-luna": {
        "label": "GPT-5.6 Luna — پولی (OpenRouter)",
        "provider": "openrouter",
        "model_id": "openai/gpt-5.6-luna",
        "free": False,
        "desc": "پولی — فقط با شارژ اعتبار OpenRouter کار می‌کند.",
    },
    "claude-sonnet": {
        "label": "Claude Sonnet 5 — پولی (OpenRouter)",
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-5",
        "free": False,
        "desc": "پولی — بهترین برای ویرایش متن. فقط با شارژ اعتبار OpenRouter.",
    },
}

# ---------------------------------------------------------------------------
# ذخیره وضعیت کاربران (مدل انتخابی + تاریخچه) در فایل JSON
# ---------------------------------------------------------------------------


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("user_state.json خراب است، از اول ساخته می‌شود.")
    return {"owner_id": None, "users": {}}


def save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except OSError as exc:
        logger.warning("ذخیره وضعیت ناموفق بود: %s", exc)


STATE = load_state()


def get_user_data(user_id: int) -> dict:
    key = str(user_id)
    users = STATE.setdefault("users", {})
    data = users.setdefault(key, {"model": DEFAULT_MODEL, "history": []})
    return data


def is_allowed(user_id: int) -> bool:
    """فقط کاربران مجاز. اگر ALLOWED_USER_IDS خالی باشد،
    اولین نفری که /start بزند صاحب بات می‌شود و بات به او قفل می‌شود."""
    if ALLOWED_USER_IDS:
        return user_id in ALLOWED_USER_IDS
    if STATE.get("owner_id") is None:
        STATE["owner_id"] = user_id
        save_state(STATE)
        logger.info("بات به کاربر %s قفل شد (اولین کاربر).", user_id)
        return True
    return user_id == STATE["owner_id"]


async def check_access(update: Update) -> bool:
    """اگر کاربر مجاز نباشد، پیام فارسی می‌دهد و False برمی‌گرداند."""
    user = update.effective_user
    if user is None or is_allowed(user.id):
        return True
    text = "⛔ این بات شخصیه و فقط صاحبش می‌تونه ازش استفاده کنه."
    if update.message:
        await update.message.reply_text(text)
    elif update.callback_query:
        await update.callback_query.answer(text, show_alert=True)
    return False


# ---------------------------------------------------------------------------
# فراخوانی مدل‌ها
# ---------------------------------------------------------------------------


async def call_openrouter(model_id: str, messages: list) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "کلید OpenRouter تنظیم نشده است. OPENROUTER_API_KEY را در .env بگذارید."
        )
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/personal-ai-bot",
        "X-Title": "Personal Educational Bot",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 429:
            raise RuntimeError(
                "سهمیه رایگان تمام شده است (خطای 429). کمی صبر کنید یا مدل را عوض کنید."
            )
        if resp.status_code == 402:
            raise RuntimeError(
                "این مدل پولی است و اعتبار OpenRouter شما کافی نیست. "
                "اول حساب OpenRouter را شارژ کنید یا یک مدل رایگان انتخاب کنید."
            )
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"پاسخ غیرمنتظره از OpenRouter: {data}")


async def call_tooken(model_id: str, messages: list) -> str:
    """فراخوانی Tooken Club — OpenAI-compatible، آدرس https://tooken.club/v1/chat/completions."""
    if not TOOKEN_API_KEY:
        raise RuntimeError(
            "کلید Tooken تنظیم نشده است. TOOKEN_API_KEY را در .env بگذارید "
            "(کلید را از داشبورد tooken.club بگیرید)."
        )
    url = f"{TOOKEN_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {TOOKEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.7,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, headers=headers, json=payload)
        if resp.status_code == 401:
            raise RuntimeError(
                "کلید Tooken نامعتبر است (خطای 401). "
                "کلید را در داشبورد tooken.club چک کنید و کلید تازه در .env بگذارید."
            )
        if resp.status_code in (402, 429):
            raise RuntimeError(
                "موجودی Tooken شما کافی نیست یا سهمیه تمام شده است "
                "(خطای 402/429). موجودی و مصرف را در داشبورد tooken.club ببینید و حساب را شارژ کنید."
            )
        if resp.status_code == 404:
            raise RuntimeError(
                f"مدل «{model_id}» در Tooken پیدا نشد (خطای 404). "
                "شناسه دقیق مدل را در tooken.club/models چک کنید "
                "و همان یک خط model_id را در MODELS در bot.py عوض کنید."
            )
        resp.raise_for_status()
        data = resp.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"پاسخ غیرمنتظره از Tooken: {data}")


async def call_gemini(model_id: str, messages: list) -> str:
    """فراخوانی مستقیم Gemini با HTTP ساده — بدون نیاز به کتابخانه اضافه."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "کلید Gemini تنظیم نشده است. GEMINI_API_KEY را در .env بگذارید "
            "(از Google AI Studio بگیرید)."
        )
    # پیام سیستم را به‌صورت system_instruction می‌فرستیم، بقیه را به contents تبدیل می‌کنیم.
    system_text = ""
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_id}:generateContent?key={GEMINI_API_KEY}"
    )
    payload: dict = {"contents": contents}
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=payload)
        if resp.status_code == 429:
            raise RuntimeError(
                "سهمیه رایگان Gemini تمام شده است (خطای 429). "
                "مدل Flash-Lite سهمیه روزانه بیشتری دارد (حدود ۵۰۰ درخواست)."
            )
        resp.raise_for_status()
        data = resp.json()
    try:
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"پاسخ غیرمنتظره از Gemini: {data}")


async def ask_model(model_key: str, messages: list) -> tuple[str, str]:
    """مدل را صدا می‌زند. اگر مدل اصلی خطا داد و رایگان بود،
    یک بار با مدل جایگزین (Gemini Flash-Lite) امتحان می‌کند.
    برمی‌گرداند: (جواب، کلید مدلی که واقعاً جواب داد)"""
    info = MODELS[model_key]
    try:
        if info["provider"] == "openrouter":
            answer = await call_openrouter(info["model_id"], messages)
        elif info["provider"] == "tooken":
            answer = await call_tooken(info["model_id"], messages)
        else:
            answer = await call_gemini(info["model_id"], messages)
        return answer, model_key
    except Exception as exc:
        logger.warning("مدل %s خطا داد: %s", model_key, exc)
        if model_key == FALLBACK_MODEL:
            raise
        # فقط برای مدل‌های رایگان، یک بار fallback را امتحان کن
        if info.get("free") and MODELS[FALLBACK_MODEL]["provider"] == "gemini":
            try:
                answer = await call_gemini(
                    MODELS[FALLBACK_MODEL]["model_id"], messages
                )
                return (
                    f"⚠️ مدل {info['label']} در دسترس نبود، "
                    f"با {MODELS[FALLBACK_MODEL]['label']} جواب می‌دهم:\n\n{answer}",
                    FALLBACK_MODEL,
                )
            except Exception:
                pass
        raise


# ---------------------------------------------------------------------------
# ابزار پیام
# ---------------------------------------------------------------------------


def split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    """پیام‌های بلندتر از سقف تلگرام را به قطعه‌های امن تقسیم می‌کند."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


async def reply_long(message, text: str) -> None:
    for chunk in split_message(text):
        await message.reply_text(chunk)


def models_keyboard() -> InlineKeyboardMarkup:
    buttons = []
    for key, info in MODELS.items():
        tag = model_tag(info)
        buttons.append(
            [InlineKeyboardButton(f"{info['label']} — {tag}", callback_data=f"model:{key}")]
        )
    return InlineKeyboardMarkup(buttons)


def model_tag(info: dict) -> str:
    """برچسب هزینه مدل: رایگان، Tooken، یا پولی OpenRouter."""
    if info["free"]:
        return "🆓 رایگان"
    if info["provider"] == "tooken":
        return "💳 Tooken — از ۵ میلیون توکن شما"
    return "💳 پولی (نیاز به شارژ OpenRouter)"


# ---------------------------------------------------------------------------
# هندلرها
# ---------------------------------------------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    user = update.effective_user
    data = get_user_data(user.id)
    model_label = MODELS[data["model"]]["label"]
    text = (
        f"سلام {user.first_name or ''}! 👋\n\n"
        "من دستیار آموزشی شخصی تو هستم.\n"
        "هر سوالی بپرس — توضیح ساده می‌دم، مثال می‌زنم و تمرین می‌دم.\n\n"
        f"🤖 مدل فعلی: {model_label}\n\n"
        "دستورها:\n"
        "/models — دیدن همه مدل‌ها و عوض کردن\n"
        "/model <نام> — عوض کردن مدل (مثلاً /model qwen-free)\n"
        "/clear — پاک کردن حافظه گفتگو\n"
        "/help — راهنما\n\n"
        "فقط بنویس، من جواب می‌دم."
    )
    await update.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    data = get_user_data(update.effective_user.id)
    model_label = MODELS[data["model"]]["label"]
    text = (
        "📖 راهنما\n\n"
        f"🤖 مدل فعلی: {model_label}\n\n"
        "• هر پیام متنی که بفرستی، با مدل فعلی جواب داده می‌شه.\n"
        "• بات فقط ۱۰ پیام آخر رو یادش می‌مونه تا مکالمه پیوسته بمونه.\n"
        "• /models — لیست مدل‌ها با دکمه\n"
        "• /model <نام> — مثال: /model deepseek-free\n"
        "• /clear — پاک کردن حافظه گفتگو\n\n"
        "مدل‌های رایگان:\n"
        "• deepseek-free — قوی‌ترین رایگان\n"
        "• qwen-free — متن عمومی\n"
        "• gemma-free — سبک و سریع\n"
        "• gemini-flash — مستقیم گوگل (۲۰ در روز)\n"
        "• gemini-lite — مستقیم گوگل (۵۰۰ در روز)\n\n"
        "مدل‌های Tooken (از ۵ میلیون توکن شما کم می‌شود):\n"
        "• tooken-deepseek — DeepSeek V4.1 Flash (پیش‌فرض) ⭐\n"
        "• tooken-deepseek-pro — DeepSeek V4 Pro\n"
        "• tooken-gpt-luna — GPT-6 Luna\n"
        "• tooken-gpt-sol — GPT-6.1 Sol\n"
        "• tooken-claude-sonnet — Claude Sonnet 5.5\n"
        "• tooken-claude-haiku — Claude Haiku 5.5\n"
        "• tooken-kimi — Kimi K3.1 Pro\n"
        "• tooken-gemini — Gemini 3.8 Flash\n\n"
        "مدل‌های پولی (نیاز به شارژ OpenRouter):\n"
        "• gpt-luna — GPT-5.6 Luna\n"
        "• claude-sonnet — Claude Sonnet 5\n\n"
        "⚠️ نکته: NotebookLM API ندارد و در بات نمی‌آید."
    )
    await update.message.reply_text(text)


async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    data = get_user_data(update.effective_user.id)
    lines = ["🤖 یک مدل انتخاب کن:\n"]
    for key, info in MODELS.items():
        mark = "✅" if key == data["model"] else "•"
        tag = model_tag(info)
        lines.append(f"{mark} {info['label']} — {tag}\n   {info['desc']}")
    await update.message.reply_text(
        "\n".join(lines), reply_markup=models_keyboard()
    )


async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    if not context.args:
        await update.message.reply_text(
            "نام مدل را هم بنویس. مثال:\n/model deepseek-free\n\n"
            "لیست کامل با /models"
        )
        return
    key = context.args[0].strip().lower()
    if key not in MODELS:
        available = ", ".join(MODELS.keys())
        await update.message.reply_text(
            f"مدل «{key}» پیدا نشد.\nمدل‌های موجود: {available}"
        )
        return
    data = get_user_data(update.effective_user.id)
    data["model"] = key
    save_state(STATE)
    info = MODELS[key]
    tag = model_tag(info)
    await update.message.reply_text(
        f"✅ مدل عوض شد: {info['label']}\n{tag}\n{info['desc']}"
    )


async def clear_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    data = get_user_data(update.effective_user.id)
    data["history"] = []
    save_state(STATE)
    await update.message.reply_text("🧹 حافظه گفتگو پاک شد. از اول شروع می‌کنیم.")


async def model_button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    if not await check_access(update):
        return
    key = query.data.replace("model:", "", 1)
    if key not in MODELS:
        await query.edit_message_text("این مدل دیگر موجود نیست.")
        return
    data = get_user_data(update.effective_user.id)
    data["model"] = key
    save_state(STATE)
    info = MODELS[key]
    tag = model_tag(info)
    await query.edit_message_text(
        f"✅ مدل عوض شد: {info['label']}\n{tag}\n{info['desc']}"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_access(update):
        return
    message = update.message
    if not message or not message.text:
        return
    user = update.effective_user
    data = get_user_data(user.id)
    model_key = data.get("model", DEFAULT_MODEL)
    if model_key not in MODELS:
        model_key = DEFAULT_MODEL

    # تاریخچه را به‌روز کن (فقط ۱۰ پیام آخر)
    history = data.setdefault("history", [])
    history.append({"role": "user", "content": message.text})
    history = history[-MAX_HISTORY:]
    data["history"] = history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history

    await context.bot.send_chat_action(
        chat_id=message.chat_id, action=ChatAction.TYPING
    )
    try:
        answer, used_model = await ask_model(model_key, messages)
    except httpx.HTTPStatusError as exc:
        logger.exception("خطای HTTP از API: %s", exc)
        await message.reply_text(
            "⚠️ خطا در ارتباط با سرویس هوش مصنوعی. "
            "کمی بعد دوباره امتحان کن. اگر ادامه داشت، مدل را با /models عوض کن."
        )
        history.pop()  # پیام ناموفق را از تاریخچه حذف کن
        data["history"] = history
        save_state(STATE)
        return
    except Exception as exc:
        logger.exception("خطا: %s", exc)
        await message.reply_text(f"⚠️ خطا: {exc}")
        history.pop()
        data["history"] = history
        save_state(STATE)
        return

    history.append({"role": "assistant", "content": answer})
    data["history"] = history[-MAX_HISTORY:]
    save_state(STATE)
    await reply_long(message, answer)


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit(
            "TELEGRAM_BOT_TOKEN تنظیم نشده است.\n"
            "فایل .env را از روی .env.example بسازید و توکن بات را از @BotFather بگیرید."
        )
    if not OPENROUTER_API_KEY and not GEMINI_API_KEY and not TOOKEN_API_KEY:
        logger.warning(
            "هیچ کلید API تنظیم نشده است (نه Tooken، نه OpenRouter، نه Gemini). "
            "بات بالا می‌آید ولی هیچ مدلی جواب نمی‌دهد. "
            "حداقل یکی را در .env بگذارید — برای مدل‌های Tooken، TOOKEN_API_KEY."
        )
    if not ALLOWED_USER_IDS and STATE.get("owner_id") is None:
        logger.warning(
            "ALLOWED_USER_IDS خالی است: اولین نفری که /start بزند صاحب بات می‌شود "
            "و بات فقط به او جواب می‌دهد. برای امنیت، آیدی عددی خودتان را در .env بگذارید."
        )

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("clear", clear_cmd))
    app.add_handler(CallbackQueryHandler(model_button, pattern=r"^model:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("بات شروع شد. مدل پیش‌فرض: %s", DEFAULT_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
