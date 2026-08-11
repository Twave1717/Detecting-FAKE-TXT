from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import urllib.request

MODEL = 'qwen3.5:4b'
SEED = 42
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 20
PRESENCE_PENALTY = 1.5
NUM_PREDICT = 1800
BASE_URL = 'http://127.0.0.1:11434'
OUT = pathlib.Path('evidence_qwen35')
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
        'id': 'P2_irrelevant_ocean_information',
        'family': 'probability_transform',
        'perturbation': 'irrelevant_information',
        'prompt': COMMON + '\n\n37개 정점의 자료를 단순화하여 단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다고 가정한다. 평균 수온은 17.8 degC, 평균 염분은 34.6, 평균 수심은 50 m이고 관측은 8월에 이루어졌다. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 밀도의 적분값이 1인지 확인하라.'
    },
    {
        'id': 'P3_question_target_change',
        'family': 'probability_transform',
        'perturbation': 'question_target_change',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따르고 Y=U^2이다. 확률 P(Y<=1)과 기대값 E[Y]를 구하라. 확률밀도함수 전체를 다시 유도할 필요는 없다.'
    },
    {
        'id': 'P4_operation_change_absolute_value',
        'family': 'probability_transform',
        'perturbation': 'operation_change',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다. 이번에는 Y=abs(U)로 정의한다. Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 밀도의 적분값이 1인지 확인하라.'
    },
    {
        'id': 'P5_missing_uniform_bounds',
        'family': 'probability_transform',
        'perturbation': 'missing_constraint',
        'prompt': '단면에 수직인 유속 U가 어떤 균일분포를 따른다. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하라. 최종 답은 <final>...</final> 안에 적어라.'
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
    {
        'id': 'W1_symbolic_current_reversal',
        'family': 'wave_dispersion',
        'perturbation': 'symbolic_sign_change',
        'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, g=9.81 m/s^2이다. 이번에는 해류가 파 진행방향과 반대여서 U=-1 m/s로 둔다. 구간 0<k<10에서 제곱식 (omega-U*k)^2=g*k*tanh(k*h)의 모든 양의 실근을 구하고, 각 근에서 sigma=omega-U*k의 부호를 적어라.'
    },
    {
        'id': 'W2_question_target_positive_branch',
        'family': 'wave_dispersion',
        'perturbation': 'question_target_change',
        'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, U=1 m/s, g=9.81 m/s^2이다. 구간 0<k<20에서 제곱식 (omega-U*k)^2=g*k*tanh(k*h)의 근 가운데, 제곱하기 전 양의 분기 omega-U*k=+sqrt(g*k*tanh(k*h))를 만족하는 k만 구하라. 제외되는 제곱식의 근이 있다면 그 값과 제외 이유도 적어라.'
    },
    {
        'id': 'W3_irrelevant_ocean_information',
        'family': 'wave_dispersion',
        'perturbation': 'irrelevant_information',
        'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, U=1 m/s, g=9.81 m/s^2이다. 현장에서는 파고 0.60 m, 수온 18.2 degC, 염분 34.7, 해수밀도 1025 kg/m^3도 함께 기록했다. 구간 0<k<20에서 omega-U*k=+sqrt(g*k*tanh(k*h))를 만족하는 파수 k를 구하고, 사용하지 않은 정보가 무엇인지 밝혀라.'
    },
]


def post_json(path: str, payload: dict, timeout: int = 1200) -> tuple[int, str, dict[str, str]]:
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={'Content-Type': 'application/json', 'User-Agent': 'ocean-math-qwen35-benchmark/1.0'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}


def get_json(path: str, timeout: int = 60) -> tuple[int, str, dict[str, str]]:
    req = urllib.request.Request(BASE_URL + path, headers={'User-Agent': 'ocean-math-qwen35-benchmark/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}

status, tags_body, tags_headers = get_json('/api/tags')
(OUT / 'ollama_tags.json').write_text(tags_body, encoding='utf-8')

records = []
for test in TESTS:
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': test['prompt']}],
        'stream': False,
        'think': True,
        'options': {
            'seed': SEED,
            'temperature': TEMPERATURE,
            'top_p': TOP_P,
            'top_k': TOP_K,
            'presence_penalty': PRESENCE_PENALTY,
            'num_predict': NUM_PREDICT,
            'num_ctx': 8192,
        },
    }
    started = dt.datetime.now(dt.timezone.utc)
    error = None
    http_status = 0
    raw = ''
    headers = {}
    try:
        http_status, raw, headers = post_json('/api/chat', payload)
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    ended = dt.datetime.now(dt.timezone.utc)
    try:
        parsed = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        parsed = None
    record = {
        'test_id': test['id'],
        'family': test['family'],
        'perturbation': test['perturbation'],
        'model_requested': MODEL,
        'seed': SEED,
        'temperature': TEMPERATURE,
        'top_p': TOP_P,
        'top_k': TOP_K,
        'presence_penalty': PRESENCE_PENALTY,
        'num_predict': NUM_PREDICT,
        'prompt': test['prompt'],
        'prompt_sha256': hashlib.sha256(test['prompt'].encode('utf-8')).hexdigest(),
        'started_at_utc': started.isoformat(),
        'ended_at_utc': ended.isoformat(),
        'elapsed_seconds': (ended - started).total_seconds(),
        'http_status': http_status,
        'response_headers': headers,
        'raw_response': raw,
        'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'parsed_response': parsed,
        'error': error,
    }
    records.append(record)
    print(test['id'], http_status, round(record['elapsed_seconds'], 1), error, flush=True)

meta = {
    'benchmark_name': 'Math for Ocean Scientists - Qwen3.5 GSM-style perturbation stress test',
    'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'runtime': 'Ollama on a fresh GitHub-hosted Ubuntu runner (CPU)',
    'model_requested': MODEL,
    'seed': SEED,
    'temperature': TEMPERATURE,
    'top_p': TOP_P,
    'top_k': TOP_K,
    'presence_penalty': PRESENCE_PENALTY,
    'num_predict': NUM_PREDICT,
    'test_count': len(TESTS),
    'response_count': len(records),
    'notes': [
        'The local model process received no ChatGPT account history, memory, or personalization.',
        'The test adapts GSM-Symbolic and GSM-Plus perturbation ideas to equations already explained in the manuscript.',
        'It is a fixed-seed illustrative stress test of one open-weight model, not a leaderboard or an estimate of commercial frontier-model accuracy.',
        'Prompts, full raw JSON responses, timestamps, model digest, and SHA-256 hashes are archived.',
    ],
}
(OUT / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'raw_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT / 'raw_records.jsonl').open('w', encoding='utf-8') as file:
    for record in records:
        file.write(json.dumps(record, ensure_ascii=False) + '\n')

md = [
    '# Qwen3.5 local benchmark raw evidence',
    '',
    f'- Model: `{MODEL}`',
    f'- Seed: `{SEED}`',
    f'- Temperature: `{TEMPERATURE}`',
    f'- Generated UTC: `{meta["created_at_utc"]}`',
    '',
]
for record in records:
    parsed = record.get('parsed_response') or {}
    message = parsed.get('message') or {}
    thinking = message.get('thinking', '')
    content = message.get('content', '')
    md.extend([
        f'## {record["test_id"]}',
        '',
        '### Prompt',
        '',
        '```text',
        record['prompt'],
        '```',
        '',
        f'Prompt SHA-256: `{record["prompt_sha256"]}`',
        '',
        '### Model thinking field (raw)',
        '',
        '```text',
        thinking,
        '```',
        '',
        '### Model answer field (raw)',
        '',
        '```text',
        content,
        '```',
        '',
        f'Response SHA-256: `{record["raw_response_sha256"]}`',
        '',
    ])
(OUT / 'raw_evidence.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
