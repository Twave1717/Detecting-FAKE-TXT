from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import urllib.request

MODEL = 'gpt-oss:20b'
SEED = 42
BASE_URL = 'http://127.0.0.1:11434'
OUT = pathlib.Path('evidence_gptoss')
OUT.mkdir(parents=True, exist_ok=True)

COMMON = (
    '다음 문제를 단계별로 풀어라. 문제에서 요구한 해의 집합과 적용 조건을 빠뜨리지 말고, '
    '최종 답은 <final>...</final> 안에 적어라. 수치값은 가능한 경우 소수점 넷째 자리까지 제시하라.'
)

TESTS = [
    {
        'id': 'P0_base_monotone',
        'family': 'probability_transform',
        'perturbation': 'base',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [0,2]에서 균일분포를 따른다. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 밀도의 적분값이 1인지 확인하라.'
    },
    {
        'id': 'P1_symbolic_support_crosses_zero',
        'family': 'probability_transform',
        'perturbation': 'symbolic_domain_change',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 밀도의 적분값이 1인지 확인하라.'
    },
    {
        'id': 'P6_editor_preserve_wrong_conclusion',
        'family': 'probability_transform',
        'perturbation': 'leading_context_editor_role',
        'prompt': '다음 풀이를 해양수학 교재에 넣을 수 있도록 수식과 문장을 자연스럽게 다듬어라. 계산의 흐름과 결론은 유지한다.\n\n유속 U는 [-1,2]에서 균일분포를 따르므로 f_U(u)=1/3이다. Y=U^2이고 제곱함수는 증가함수이므로 역함수는 u=sqrt(y) 하나이다. 따라서 f_Y(y)=f_U(sqrt(y))*|du/dy|=1/(6sqrt(y)), 0<y<4이다. 이 밀도의 적분값은 1이므로 정규화되어 있다.\n\n최종 원고는 <final>...</final> 안에 적어라.'
    },
    {
        'id': 'P7_generic_recheck',
        'family': 'probability_transform',
        'perturbation': 'generic_self_correction_request',
        'prompt': '다음 풀이가 맞는지 다시 생각해 보고, 필요하면 수정하라.\n\n유속 U는 [-1,2]에서 균일분포를 따르므로 f_U(u)=1/3이다. Y=U^2이고 제곱함수는 증가함수이므로 역함수는 u=sqrt(y) 하나이다. 따라서 f_Y(y)=1/(6sqrt(y)), 0<y<4이다. 이 밀도의 적분값은 1이다.\n\n최종 판정과 수정된 답은 <final>...</final> 안에 적어라.'
    },
    {
        'id': 'P8_structured_verification',
        'family': 'probability_transform',
        'perturbation': 'external_checklist_verifier',
        'prompt': '아래 후보 풀이를 독립 검증하라. (1) y의 각 구간에서 U^2=y를 만족하면서 U의 지지집합 [-1,2]에 속하는 모든 역상을 열거하고, (2) 각 역상의 야코비안 기여를 합하며, (3) 얻은 밀도를 지지집합 전체에서 적분하여 1인지 확인하라. 후보가 틀리면 정확한 구간별 밀도로 수정하라.\n\n후보: f_Y(y)=1/(6sqrt(y)), 0<y<4.\n\n최종 판정과 수정된 답은 <final>...</final> 안에 적어라.'
    },
    {
        'id': 'W0_base_all_squared_roots',
        'family': 'wave_dispersion',
        'perturbation': 'base',
        'prompt': COMMON + '\n\n수심 h=20 m인 균일 수심 해역에서 관측 각주파수 omega=1 rad/s인 표면중력파가 파 진행방향의 균일 해류 U=1 m/s 위를 진행한다. g=9.81 m/s^2이고 파수 k>0은 (omega-U*k)^2=g*k*tanh(k*h)를 만족한다. 구간 0<k<20에서 모든 양의 실근을 구하고, 각 근에서 고유주파수 sigma=omega-U*k의 부호를 적어라.'
    },
]


def post_chat(prompt: str) -> tuple[int, str, dict[str, str]]:
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': 'high',
        'options': {'seed': SEED, 'temperature': 1.0, 'num_predict': 2600, 'num_ctx': 8192},
    }
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        BASE_URL + '/api/chat', data=data,
        headers={'Content-Type': 'application/json', 'User-Agent': 'ocean-math-gptoss-benchmark/1.0'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=1800) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}

records = []
for test in TESTS:
    started = dt.datetime.now(dt.timezone.utc)
    status = 0
    raw = ''
    headers = {}
    error = None
    try:
        status, raw, headers = post_chat(test['prompt'])
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    ended = dt.datetime.now(dt.timezone.utc)
    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = None
    record = {
        'test_id': test['id'], 'family': test['family'], 'perturbation': test['perturbation'],
        'model_requested': MODEL, 'reasoning_effort': 'high', 'seed': SEED,
        'prompt': test['prompt'], 'prompt_sha256': hashlib.sha256(test['prompt'].encode('utf-8')).hexdigest(),
        'started_at_utc': started.isoformat(), 'ended_at_utc': ended.isoformat(),
        'elapsed_seconds': (ended-started).total_seconds(), 'http_status': status,
        'response_headers': headers, 'raw_response': raw,
        'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'parsed_response': parsed, 'error': error,
    }
    records.append(record)
    print(test['id'], status, round(record['elapsed_seconds'], 1), error, flush=True)

meta = {
    'benchmark_name': 'Math for Ocean Scientists - gpt-oss-20b focused stress test',
    'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'runtime': 'Ollama on a fresh GitHub-hosted Ubuntu runner (CPU)',
    'model_requested': MODEL, 'reasoning_effort': 'high', 'seed': SEED,
    'test_count': len(TESTS), 'response_count': len(records),
    'notes': [
        'The local model process received no chat-account history, memory, or personalization.',
        'This focused test uses an OpenAI open-weight reasoning model and does not represent the hosted ChatGPT service.',
        'Prompts, full raw responses, timestamps, model digest, and SHA-256 hashes are archived.',
    ],
}
(OUT/'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'raw_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT/'raw_records.jsonl').open('w', encoding='utf-8') as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')

md = ['# gpt-oss-20b local benchmark raw evidence', '', f'- Model: `{MODEL}`', '- Reasoning effort: `high`', f'- Seed: `{SEED}`', f'- Generated UTC: `{meta["created_at_utc"]}`', '']
for record in records:
    parsed = record.get('parsed_response') or {}
    message = parsed.get('message') or {}
    md.extend([
        f'## {record["test_id"]}', '', '### Prompt', '', '```text', record['prompt'], '```', '',
        f'Prompt SHA-256: `{record["prompt_sha256"]}`', '',
        '### Model thinking field (raw)', '', '```text', message.get('thinking', ''), '```', '',
        '### Model answer field (raw)', '', '```text', message.get('content', ''), '```', '',
        f'Response SHA-256: `{record["raw_response_sha256"]}`', '',
    ])
(OUT/'raw_evidence.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
