FROM python:3.12-slim

WORKDIR /app

# System deps for pdfminer/pypdf/python-docx
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libxml2-dev \
        libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy project and install from local source (this fork)
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
RUN uv pip install --system --no-cache-dir .

# Default ports: MCP SSE on 8000, optional WebUI on 8765
EXPOSE 8000 8765

# Default env — override at runtime
ENV QDRANT_URL="http://qdrant:6333"
ENV EMBEDDING_PROVIDER="fastembed"
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
ENV QDRANT_ENABLE_COLLECTION_MANAGEMENT="true"
ENV QDRANT_ENABLE_DYNAMIC_EMBEDDING_MODELS="true"
ENV QDRANT_ENABLE_RESOURCES="true"
ENV QDRANT_DEFAULT_VECTOR_SIZE="384"
ENV FASTMCP_PORT="8000"

CMD ["mcp-server-qdrant", "--transport", "sse"]
