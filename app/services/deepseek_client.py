from __future__ import annotations

import os
from typing import List, Optional

import requests


class DeepSeekClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url or os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
        self.model = model or os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        self.timeout = timeout or int(os.getenv('DEEPSEEK_TIMEOUT', '60'))

    def complete(self, messages: List[dict], temperature: float = 0.2, model: Optional[str] = None) -> str:
        if not self.api_key:
            raise RuntimeError('DEEPSEEK_API_KEY is not set.')

        endpoint = self.base_url.rstrip('/')
        if not endpoint.endswith('/chat/completions'):
            endpoint = endpoint + '/chat/completions'

        response = requests.post(
            endpoint,
            headers={
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
            },
            json={
                'model': model or self.model,
                'messages': messages,
                'temperature': temperature,
                'stream': False,
            },
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f'DeepSeek API error {response.status_code}: {response.text[:500]}')

        data = response.json()
        return data['choices'][0]['message']['content']
