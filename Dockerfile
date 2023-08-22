# Select base image: https://catalog.ngc.nvidia.com/containers
# https://qiita.com/k_ikasumipowder/items/32bf0bc781cbbdfa2edb#%E8%A4%87%E6%95%B0%E3%81%AE%E3%83%90%E3%83%BC%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%AE%E3%82%B3%E3%83%B3%E3%83%86%E3%83%8A%E3%81%AE%E3%83%86%E3%82%B9%E3%83%88
FROM nvidia/cudagl:10.2-devel-ubuntu18.04

# https://zenn.dev/flyingbarbarian/scraps/1275681132babd
ENV DEBIAN_FRONTEND noninteractive

# install zsh https://github.com/ohmyzsh/ohmyzsh#prerequisites
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    wget curl git zsh
SHELL ["/bin/zsh", "-c"]
RUN curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh | zsh

# install pyenv
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | zsh && \
    echo '' >> /root/.zshrc && \
    echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /root/.zshrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /root/.zshrc && \
    echo 'eval "$(pyenv init --path)"' >> /root/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.zshrc && \
    source /root/.zshrc && \
    pyenv install 3.8.0 && \
    pyenv global 3.8.0 && \
    pip install -U pip

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /root
CMD ["zsh"]

