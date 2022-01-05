# Base image
FROM anibali/pytorch:1.8.1-cuda11.1

WORKDIR /

#run with e.g. docker run --rm --name test -v $PWD/pytorch_docker.py:/pytorch_docker.py torch:latest python3 pytorch_docker.py