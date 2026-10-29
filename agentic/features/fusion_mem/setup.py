from setuptools import setup, find_packages

setup(
    name="fusion_mem",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.115.8",
        "uvicorn==0.34.0",
        "pydantic==2.10.4",
        "python-dotenv==1.0.1",
        "psycopg>=3.2.8",
        "openai>=1.90.0",
        "protobuf>=5.29.0,<6.0.0",
        "langchain-neo4j>=0.4.0",
        "langchain-aws>=0.2.23",
        "langchain-memgraph>=0.1.0",
        "chromadb>=0.4.24",
        "pytz",
        "requests",
    ],
    entry_points={
        'console_scripts': [
            'fusion_mem = fusion_mem.cli:main',
        ],
    },
    author="Your Name",
    description="Fusion Mem: Server and Client CLI",
    python_requires='>=3.8',
)