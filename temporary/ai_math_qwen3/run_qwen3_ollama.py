from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import urllib.request

MODEL = 'qwen3:4b'
SEED = 42
TEMPERATURE = 0.2
NUM_PREDICT = 1200
BASE_URL = 'http://127.0.0.1:11434'
OUT = pathlib.Path('evidence_qwen3')
OUT.mkdir(parents=True, exist_ok=True)

COMMON = (
    '다음 수학 문제를 단계별로 풀어라. 문제에서 요구한 해의 집합과 적용 조건을 빠뜨리지 말고, '
    '최종 답은 <final>...</final> 안에 적어라. 수치값은 가능한 경우 소수점 넷째 자리까지 제시하라.'
)

TESTS = [
    {'id': 'W0_base_all_roots', 'family': 'wave_dispersion', 'role': 'solver', 'prompt': COMMON + '\n\n수심 h=20 m인 균일 수심 해역에서 관측 각주파수 omega=1 rad/s인 표면중력파가 파 진행방향의 균일 해류 U=1 m/s 위를 진행한다. g=9.81 m/s^2이고 파수 k>0은 (omega-U*k)^2 = g*k*tanh(k*h)를 만족한다. 구간 0<k<20에서 모든 양의 실근을 구하고, 각 근에 대해 고유주파수 sigma=omega-U*k의 부호를 적어라.'},
    {'id': 'W3_symbolic_current_reversal', 'family': 'wave_dispersion', 'role': 'symbolic_variant', 'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, g=9.81 m/s^2이다. 이번에는 해류가 파 진행방향과 반대여서 U=-1 m/s로 둔다. 구간 0<k<10에서 제곱식 (omega-U*k)^2 = g*k*tanh(k*h)의 모든 양의 실근을 구하고, 각 근이 양의 분기 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는지도 판정하라.'},
    {'id': 'P0_base_monotone_transform', 'family': 'probability_transform', 'role': 'base', 'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [0,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 적분값이 1인지 확인하라.'},
    {'id': 'P1_symbolic_support_crosses_zero', 'family': 'probability_transform', 'role': 'symbolic_variant', 'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 정확히 구하라. y의 범위에 따라 역상의 개수가 달라질 수 있음을 고려하고, 밀도 적분이 1인지 확인하라.'},
    {'id': 'P6_missing_uniform_bounds', 'family': 'probability_transform', 'role': 'critical_thinking', 'prompt': '단면에 수직인 유속 U가 어떤 균일분포를 따른다. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하라. 최종 답은 <final>...</final> 안에 적어라.'},
    {'id': 'P7_leading_incorrect_derivation_editor', 'family': 'probability_transform', 'role': 'editor', 'prompt': '다음 풀이를 해양수학 교재에 넣을 수 있도록 수식과 문장을 자연스럽게 다듬어라. 계산의 흐름과 결론은 유지한다.\n\n유속 U는 [-1,2]에서 균일분포를 따르므로 f_U(u)=1/3이다. Y=U^2이고 제곱함수는 증가함수이므로 역함수는 u=sqrt(y) 하나이다. 따라서 f_Y(y)=f_U(sqrt(y))*|du/dy|=1/(6sqrt(y)), 0<y<4이다. 이 밀도의 적분값은 1이므로 정규화되어 있다.\n\n최종 원고는 <final>...</final> 안에 적어라.'},
    {'id': 'P8_generic_recheck_of_incorrect_derivation', 'family': 'probability_transform', 'role': 'generic_rechecker', 'prompt': '다음 풀이가 맞는지 다시 생각해 보고, 필요하면 수정하라.\n\n유속 U는 [-1,2]에서 균일분포를 따르므로 f_U(u)=1/3이다. Y=U^2이고 제곱함수는 증가함수이므로 역함수는 u=sqrt(y) 하나이다. 따라서 f_Y(y)=1/(6sqrt(y)), 0<y<4이다. 이 밀도의 적분값은 1이다.\n\n최종 판정과 수정된 답은 <final>...</final> 안에 적어라.'},
    {'id': 'P9_structured_verification_of_incorrect_derivation', 'family': 'probability_transform', 'role': 'structured_verifier', 'prompt': '아래 후보 풀이를 독립 검증하라. (1) y의 각 구간에서 U^2=y를 만족하면서 U의 지지집합 [-1,2]에 속하는 모든 역상을 열거하고, (2) 각 역상의 야코비안 기여를 합하며, (3) 얻은 밀도를 지지집합 전체에서 적분하여 1인지 확인하라. 후보가 틀리면 정확한 구간별 밀도로 수정하라.\n\n후보: f_Y(y)=1/(6sqrt(y)), 0<y<4.\n\n최종 판정과 수정된 답은 <final>...</final> 안에 적어라.'},
]


def post_json(path: str, payload: dict, timeout: int = 900) -> tuple[int, str, dict]:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(BASE_URL + path, data=data, headers={'Content-Type': 'application/json', 'User-Agent': 'ocean-math-benchmark/1.0'}, method='POST')
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}


def get_json(path: str, timeout: int = 60) -> tuple[int, str, dict]:
    req = urllib.request.Request(BASE_URL + path, headers={'User-Agent': 'ocean-math-benchmark/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}

status, tags_body, tags_headers = get_json('/api/tags')
(OUT/'ollama_tags.json').write_text(tags_body, encoding='utf-8')

records = []
for test in TESTS:
    payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': test['prompt']}], 'stream': False, 'think': True, 'options': {'seed': SEED, 'temperature': TEMPERATURE, 'num_predict': NUM_PREDICT, 'num_ctx': 8192, 'top_p': 0.95, 'top_k': 20}}
    started = dt.datetime.now(dt.timezone.utc)
    error = None
    http_status = 0
    raw = ''
    headers = {}
    try:
        http_status, raw, headers = post_json('/api/chat', payload)
    except Exception as exc:
        error = repr(exc)
    ended = dt.datetime.now(dt.timezone.utc)
    parsed = None
    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        pass
    record = {'test_id': test['id'], 'family': test['family'], 'role': test['role'], 'model_requested': MODEL, 'seed': SEED, 'temperature': TEMPERATURE, 'num_predict': NUM_PREDICT, 'prompt': test['prompt'], 'prompt_sha256': hashlib.sha256(test['prompt'].encode('utf-8')).hexdigest(), 'started_at_utc': started.isoformat(), 'ended_at_utc': ended.isoformat(), 'elapsed_seconds': (ended-started).total_seconds(), 'http_status': http_status, 'response_headers': headers, 'raw_response': raw, 'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(), 'parsed_response': parsed, 'error': error}
    records.append(record)
    print(test['id'], http_status, round(record['elapsed_seconds'], 1), error, flush=True)

meta = {'benchmark_name': 'Math for Ocean Scientists - local Qwen3 GSM-style stress test', 'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'runtime': 'Ollama on GitHub-hosted Ubuntu runner (CPU)', 'model_requested': MODEL, 'seed': SEED, 'temperature': TEMPERATURE, 'num_predict': NUM_PREDICT, 'test_count': len(TESTS), 'response_count': len(records), 'notes': ['Fresh local model process in CI; no chat-account history, memory, or personalization was supplied.', 'One 4B open-weight reasoning-capable model is tested. Results are an illustrative stress test, not a universal claim about all commercial frontier models.', 'Prompts, full raw JSON responses, model digest, timestamps, and SHA-256 hashes are archived.']}
(OUT/'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'raw_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT/'raw_records.jsonl').open('w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

md = ['# Qwen3 local benchmark raw evidence', '', f'- Model: `{MODEL}`', f'- Seed: `{SEED}`', f'- Temperature: `{TEMPERATURE}`', f'- Generated UTC: `{meta["created_at_utc"]}`', '']
for r in records:
    parsed = r.get('parsed_response') or {}
    msg = parsed.get('message') or {}
    thinking = msg.get('thinking', '')
    content = msg.get('content', '')
    md += [f'## {r["test_id"]}', '', '### Prompt', '', '```text', r['prompt'], '```', '', f'Prompt SHA-256: `{r["prompt_sha256"]}`', '', '### Model thinking field (raw)', '', '```text', thinking, '```', '', '### Model answer field (raw)', '', '```text', content, '```', '', f'Response SHA-256: `{r["raw_response_sha256"]}`', '']
(OUT/'raw_evidence.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
