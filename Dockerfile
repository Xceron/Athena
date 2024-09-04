FROM python:3.12

ADD --chmod=755 https://astral.sh/uv/install.sh /install.sh
RUN /install.sh && rm /install.sh

WORKDIR /app
COPY requirements.txt .

# uv
RUN /root/.cargo/bin/uv pip install --system --no-cache -r requirements.txt
COPY . .
EXPOSE 5000

# Python logger fix
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]