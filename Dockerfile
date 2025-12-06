# Minimal CPU image for Causal Copilot notebooks

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

# 1) System packages: graphviz for pydot visualizations
RUN apt-get update && apt-get install -y --no-install-recommends \
    graphviz \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2) Workdir inside container
WORKDIR /workspace

# 3) Install Python dependencies
COPY requirements_notebooks.txt .
RUN pip install --no-cache-dir -r requirements_notebooks.txt

# 4) Copy the repo into the image
COPY . .

# 5) Default: drop into a shell
CMD ["/bin/bash"]
