FROM python:3.12-slim
WORKDIR /srv
COPY . /srv
CMD ["python", "mcp_server.py", "--seed", "3", "--transport", "http", "--host", "0.0.0.0", "--port", "8000", "--result-file", "/srv/result.json"]
