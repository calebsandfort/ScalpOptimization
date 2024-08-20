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
    libopenblas-dev \
    software-properties-common \
    git


    # Add the GitHub CLI repository
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
# RUN apt-add-repository https://cli.github.com/packages   

# # Update package lists again after adding the new repository
# RUN apt-get update
# # Install git and gh using apt (as root)
# RUN apt-get install -y \
#     git

# Install necessary tools for downloading and extracting the binary
RUN apt-get update && apt-get install -y \
    curl \
    unzip

# Download the GitHub CLI binary for amd64 architecture
RUN curl -LO https://github.com/cli/cli/releases/download/v2.55.0/gh_2.55.0_linux_amd64.tar.gz 

# Extract the binary
RUN tar -xzvf gh_2.55.0_linux_amd64.tar.gz 

# Make the binary executable
RUN chmod +x gh_2.55.0_linux_amd64/bin/gh

# Move the binary to a location in the PATH (e.g., /usr/local/bin)
RUN mv gh_2.55.0_linux_amd64/bin/gh /usr/local/bin/gh

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
