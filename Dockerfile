FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Runtime writable directories. They are created and owned by the non-root user
# in the image, so the named volumes mounted over them at `docker compose up`
# inherit that ownership and stay writable without running as root.
RUN mkdir -p /app/logs /app/models /app/shared_files \
    && groupadd --system appuser \
    && useradd --system --gid appuser --home-dir /app appuser \
    && chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1

USER appuser

EXPOSE 8899

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8899"]
