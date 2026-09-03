FROM node:20-slim

# Python + torch (CPU wheel) + pandas for the IAC bridge (server-python/).
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# +cpu build only exists on the pytorch index; deps resolve from PyPI.
RUN pip install --no-cache-dir "torch==2.9.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir pandas

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

COPY server.js ./
COPY server-python ./server-python
COPY public ./public

EXPOSE 3000
CMD ["node", "server.js"]
