from __future__ import annotations

from pathlib import Path
from typing import Optional

from app.schemas.chat import ChatRequest
from app.services.deepseek_client import DeepSeekClient
from app.services.result_summarizer import build_llm_context
from app.services.session_store import SessionStore


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = PROJECT_ROOT / 'prompts' / 'deepseek_tsad_system_prompt.md'


class ChatService:
    def __init__(
        self,
        store: SessionStore,
        deepseek_client: Optional[DeepSeekClient] = None,
        prompt_path: Path = DEFAULT_PROMPT,
    ):
        self.store = store
        self.deepseek_client = deepseek_client or DeepSeekClient()
        self.system_prompt = prompt_path.read_text(encoding='utf-8')

    def answer(self, request: ChatRequest) -> dict:
        record = self.store.get_result(request.result_id)
        if record is None:
            raise ValueError(f'Unknown result_id: {request.result_id}')

        context = build_llm_context(record, request.window_index)
        history = self.store.get_history(request.result_id, limit=8)
        current_message = {
            'role': 'user',
            'content': (
                '以下是当前检测层输出的结构化证据。'
                '如果用户问题与本次检测结果有关，请基于这些证据回答；'
                '如果用户询问通用时序异常检测知识，可以结合领域知识回答。\n\n'
                f'```json\n{context}\n```\n\n'
                f'用户问题：{request.message}'
            ),
        }
        messages = [{'role': 'system', 'content': self.system_prompt}] + history + [current_message]
        answer = self.deepseek_client.complete(messages, model=request.model)

        self.store.append_message(request.result_id, 'user', request.message)
        self.store.append_message(request.result_id, 'assistant', answer)
        return {
            'answer': answer,
            'history': self.store.get_history(request.result_id, limit=20),
        }
