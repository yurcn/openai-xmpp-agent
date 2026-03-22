FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

RUN useradd -r -u 10001 -g users appuser

COPY requirements.txt .
RUN python -m pip install --upgrade pip && python -m pip install -r requirements.txt

COPY --chown=appuser:users openai-xmpp-agent.py ./openai-xmpp-agent.py

USER appuser

# Pass secrets at runtime:
# XMPP_JID, XMPP_PASSWORD, OPENAI_API_KEY, OPENAI_MODEL, OPENAI_PROMPT_ID

CMD ["python", "openai-xmpp-agent.py"]
