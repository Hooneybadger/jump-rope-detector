FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /srv/app app

WORKDIR /srv/app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.2.1 setuptools==83.0.0 \
    && python -m pip install --no-cache-dir --requirement requirements.txt \
    && python -m pip uninstall --yes pip setuptools

COPY --chown=app:app app ./app
COPY --chown=app:app static ./static
COPY --chown=app:app basic_jump ./basic_jump
COPY --chown=app:app alternating_jump ./alternating_jump
COPY --chown=app:app double_jump ./double_jump
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app alembic.ini ./

USER app
EXPOSE 8000
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 --proxy-headers --forwarded-allow-ips=*"]
