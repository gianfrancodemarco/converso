import os


def pytest_sessionstart(session):
    os.environ["OPENAI_API_KEY"] = "sk-..."
