FROM python:3.11-slim-buster

RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
COPY Makefile README.md  ./
COPY src/ src/

RUN python -m pip install --upgrade pip && \ 
	python -m pip install -r requirements.txt --no-cache-dir && \
	python -m pip install . --no-cache-dir 


WORKDIR /

EXPOSE 6666

CMD ["panel", "serve", "src/graphs/app/main.py", "--address", "0.0.0.0", "--port", "6666",  "--allow-websocket-origin", "*", "--index", "main"]
