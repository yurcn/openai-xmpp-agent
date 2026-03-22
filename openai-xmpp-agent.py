import asyncio
import os
import logging
import sys
import re
import tempfile
import aiohttp
import base64

from urllib.parse import urlparse
from openai import AsyncOpenAI
from slixmpp import ClientXMPP

logging.basicConfig(level=logging.DEBUG)

AUDIO_URL_RE = re.compile(r"https?://\S+\.(m4a|mp3|wav|webm)(\?\S+)?$", re.IGNORECASE)
IMAGE_URL_RE = re.compile(r"https?://\S+\.(jpg|jpeg|png)(\?\S+)?$", re.IGNORECASE)

class ChatGPTBot(ClientXMPP):
    def __init__(self, jid, password, model, api_key, prompt_id):
        super().__init__(jid, password)
        self.register_plugin("xep_0030")
        self.register_plugin("xep_0115")
        self.register_plugin("xep_0085")
        self.model = model
        self.prompt_id = prompt_id

        self.add_event_handler("session_start", self.session_start)
        self.add_event_handler("message", self.message)
        self.add_event_handler("failed_auth", lambda _: sys.exit("Auth failed"))

        self.aclient = AsyncOpenAI(api_key=api_key)

    async def session_start(self, event):
        self.send_presence()
        await self.get_roster()
        logging.info("XMPP bot is online")

    async def message(self, msg):
        if msg["type"] in ("chat", "normal"):
            user_input = str(msg["body"])
            logging.info(user_input)

            to_jid = msg["from"].bare
            stop = asyncio.Event()
            typing_task = asyncio.create_task(self._typing_pulse(to_jid, stop))

            try:
                if AUDIO_URL_RE.search(user_input):
                    reply_text = await self.handle_audio_url(user_input)
                elif IMAGE_URL_RE.search(user_input):
                    text = await self.ocr_image_url(user_input)
                    reply_text = await self.ask_gpt(text)
                else:
                    reply_text = await self.ask_gpt(user_input)
            finally:
                stop.set()
                typing_task.cancel()
                self._typing_off(to_jid)

            msg.reply(reply_text).send()

    async def handle_audio_url(self, url: str) -> str:
        try:
            audio_path = await self.download_audio(url)
            transcript = await self.transcribe_audio(audio_path)
            return await self.ask_gpt(transcript)
        except Exception as e:
            return f"Audio processing error: {e}"

    async def download_to_temp(self, url: str, suffix: str, max_bytes: int) -> str:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as r:
                r.raise_for_status()
                data = await r.read()

        if len(data) > max_bytes:
            raise ValueError(f"File is larger than {max_bytes} bytes")

        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return path

    async def download_audio(self, url: str) -> str:
        return await self.download_to_temp(url, suffix=".m4a", max_bytes=25 * 1024 * 1024)

    async def download_image(self, url: str, ext: str) -> str:
        return await self.download_to_temp(url, suffix=f".{ext}", max_bytes=25 * 1024 * 1024)

    def _ext_from_url(self, url: str) -> str:
        p = urlparse(url).path
        ext = os.path.splitext(p)[1].lstrip(".").lower()
        if ext in ("jpg", "jpeg", "png"):
            return ext
        return "jpg"

    def _mime_from_ext(self, ext: str) -> str:
        ext = ext.lower()
        if ext in ("jpg", "jpeg"):
            return "image/jpeg"
        if ext == "png":
            return "image/png"
        return "application/octet-stream"

    async def transcribe_audio(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                tr = await self.aclient.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=f,
                    response_format="text",
                )
            return getattr(tr, "text", str(tr)).strip()
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def ocr_image_url(self, url: str) -> str:
        ext = self._ext_from_url(url)
        path = await self.download_image(url, ext)
        try:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            mime = self._mime_from_ext(ext)

            resp = await self.aclient.responses.create(
                model=self.model,
                temperature=0,
                max_output_tokens=1024,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",
                             "text": "Extract ALL readable text from this image. Output ONLY the extracted text."},
                            {"type": "input_image", "image_url": f"data:{mime};base64,{b64}"},
                        ],
                    }
                ],
            )
            text = (resp.output_text or "").strip()
            if not text:
                raise ValueError("No text detected in image")
            return text
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    async def ask_gpt(self, user_input: str) -> str:
        try:
            resp = await self.aclient.responses.create(
                model=self.model,
                prompt={
                    "id": self.prompt_id,
                    "variables": {"text": user_input},
                },
                temperature=0,
                max_output_tokens=1024,
            )
            return (resp.output_text or "").strip()
        except Exception as e:
            return f"OpenAI error: {str(e)}"

    async def _typing_pulse(self, to_jid: str, stop_event: asyncio.Event, interval: float = 8.0):
        try:
            m = self.make_message(mto=to_jid, mtype="chat")
            m["chat_state"] = "composing"
            m.send()

            while not stop_event.is_set():
                await asyncio.sleep(interval)
                m = self.make_message(mto=to_jid, mtype="chat")
                m["chat_state"] = "composing"
                m.send()
        except Exception:
            pass

    def _typing_off(self, to_jid: str):
        try:
            m = self.make_message(mto=to_jid, mtype="chat")
            m["chat_state"] = "active"
            m.send()
        except Exception:
            pass

async def main():
    JID = os.getenv("XMPP_JID")
    PASSWORD = os.getenv("XMPP_PASSWORD")
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4.1-mini"

    OPENAI_PROMPT_ID = os.getenv("OPENAI_PROMPT_ID")

    if not all([JID, PASSWORD, OPENAI_KEY, OPENAI_PROMPT_ID]):
        missing = [k for k, v in {
            "XMPP_JID": JID,
            "XMPP_PASSWORD": PASSWORD,
            "OPENAI_API_KEY": OPENAI_KEY,
            "OPENAI_PROMPT_ID": OPENAI_PROMPT_ID,
        }.items() if not v]
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")

    xmpp = ChatGPTBot(JID, PASSWORD, OPENAI_MODEL, OPENAI_KEY, OPENAI_PROMPT_ID)
    await xmpp.connect()
    await xmpp.disconnected


if __name__ == "__main__":
    asyncio.run(main())
