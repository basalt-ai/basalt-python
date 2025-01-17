import os

build = os.getenv("BUILD", "production")

config = {
    'api_url': 'http://localhost:3001' if build == 'development' else 'https://api.getbasalt.ai',
    'sdk_version': '0.0.1',
    'sdk_type': 'python',
}

