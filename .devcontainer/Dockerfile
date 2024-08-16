ARG VARIANT="3.10-7.3.16-slim-bullseye"
FROM pypy:${VARIANT}

# System updates and installations
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    gfortran \
    pkg-config \
    cmake \
    libblas-dev \
    libopenblas-dev

# Copy project files (replace with your actual copy command)
RUN useradd -ms /bin/bash -u 1052718 csandfort

# Copy requirements and install as the csandfort user
USER csandfort 
WORKDIR /home/csandfort
COPY requirements.txt /home/csandfort/requirements.txt
RUN pip3 install --user -r requirements.txt

# Switch back to root for other installations (if needed)
USER root
# ... (other installation commands, if any)

# Switch back to the csandfort user for running the application
USER csandfort
# ... (command to run your application)
