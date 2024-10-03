# PMLDL Assignment 1: Deployment

| Name         | Innomail                       | Group     |
| ------------ | ------------------------------ | --------- |
| Egor Machnev | e.machnev@innopolis.university | B22-AI-02 |

## Overview

This repository contains code for Assignment 1 of the Practical Machine Learning
and Deep Learning course. The primary objective of this project is to build a
machine learning model that predicts whether a person in an image is a blonde or
not. The model is based on the work completed in Lab 2, utilizing the PyTorch
library for implementation.

The project consists of two main services:

1. **API** - Built using FastAPI.
2. **Web Application** - Developed with Streamlit.

This code is compatible with Python version 3.12.

## Dependencies

To get started with the project, you need to install the required dependencies
using Poetry. Follow the steps below:

1. Ensure that you have Poetry installed. If not, you can follow the
   installation instructions on the
   [Poetry website](https://python-poetry.org/docs/#installation).
2. Once Poetry is installed, navigate to the project directory and run the
   following command:

   ```bash
   poetry install
   ```

   This command will install all the necessary dependencies specified in the
   pyproject.toml file.

## Training the Model

To train the machine learning model, follow these steps:

1. Navigate to the `code/models` directory:

   ```bash
   cd code/models
   ```

2. Run the following command:

   ```bash
   poetry run python train.py
   ```

   The training process will begin, and once completed, the model will be ready
   for deployment.

## Running Docker Services

To run the API and Web Application services using Docker, follow these steps:

1. Navigate to the `code/deployment` directory:

   ```bash
   cd code/deployment
   ```

2. Build and start the Docker services by executing the following command:

   ```bash
   docker-compose up --build
   ```

   This command will build the Docker images and start the services. The API
   will be accessible at `http://localhost:8000` and the Web Application at
   `http://localhost:8501`.

## Conclusion

This README provides a quick start guide to setting up and running the machine
learning model and the associated services. By following these instructions, you
should be able to train the model and deploy the API and web application
successfully.
