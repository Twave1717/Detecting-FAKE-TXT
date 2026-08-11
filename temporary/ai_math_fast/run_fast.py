from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.parse
import urllib.request

MODEL = 'openai-fast'
SEED = 42
BASE_URL = 'https://text.pollinations.ai'
OUT = pathlib.Path('evidence_fast')
OUT.mkdir(parents=True, exist_ok=True)

COMMON = (
    '다음 수학 문제를 단계별로 풀어라. 문제에서 요구한 해의 집합과 적용 조건을 빠뜨리지 말고, '
    '최종 답은 <final>...</final> 안에 적어라. 수치값은 가능한 경우 소수점 넷째 자리까지 제시하라.'
)

TESTS = [
    {
        'id': 'W0_base_all_roots', 'family': 'wave_dispersion', 'perturbation': 'base',
        'prompt': COMMON + '\n\n수심 h=20 m인 균일 수심 해역에서 관측 각주파수 omega=1 rad/s인 표면중력파가 파 진행방향의 균일 해류 U=1 m/s 위를 진행한다. g=9.81 m/s^2이고 파수 k>0은 (omega-U*k)^2 = g*k*tanh(k*h)를 만족한다. 구간 0<k<20에서 모든 양의 실근을 구하고, 각 근에 대해 고유주파수 sigma=omega-U*k의 부호를 적어라.'
    },
    {
        'id': 'W1_target_positive_branch', 'family': 'wave_dispersion', 'perturbation': 'question_target',
        'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, U=1 m/s, g=9.81 m/s^2이다. 구간 0<k<20에서 제곱식 (omega-U*k)^2 = g*k*tanh(k*h)의 근 가운데, 제곱하기 전 양의 분기 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는 k만 구하라. 제외되는 제곱식의 근이 있다면 그 값과 제외 이유도 적어라.'
    },
    {
        'id': 'W3_symbolic_current_reversal', 'family': 'wave_dispersion', 'perturbation': 'symbolic_sign_change',
        'prompt': COMMON + '\n\n수심 h=20 m, omega=1 rad/s, g=9.81 m/s^2이다. 이번에는 해류가 파 진행방향과 반대여서 U=-1 m/s로 둔다. 구간 0<k<10에서 제곱식 (omega-U*k)^2 = g*k*tanh(k*h)의 모든 양의 실근을 구하고, 각 근이 양의 분기 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는지도 판정하라.'
    },
    {
        'id': 'P0_base_monotone_transform', 'family': 'probability_transform', 'perturbation': 'base',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [0,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 적분값이 1인지 확인하라.'
    },
    {
        'id': 'P1_symbolic_support_crosses_zero', 'family': 'probability_transform', 'perturbation': 'symbolic_interval_change',
        'prompt': COMMON + '\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 정확히 구하라. y의 범위에 따라 역상의 개수가 달라질 수 있음을 고려하고, 밀도 적분이 1인지 확인하라.'
    },
    {
        'id': 'P5_misleading_claim', 'family': 'probability_transform', 'perturbation': 'misleading_statement',
        'prompt': COMMON + '\n\n유속 U가 [-1,2]에서 균일분포를 따르고 Y=U^2이다. 한 연구자는 제곱함수는 증가함수이므로 U=sqrt(Y) 하나만 역함수로 쓰면 된다고 주장한다. 이 주장을 검토하고, 틀렸다면 정확한 f_Y(y)와 지지집합을 제시하라.'
    },
    {
        'id': 'D0_base_midlatitude', 'family': 'daylength', 'perturbation': 'base',
        'prompt': COMMON + '\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용한다. phi는 위도(rad), delta는 태양 적위(rad)이다. phi=35.18 deg, delta=23.44 deg일 때 D를 시간 단위로 계산하라. 대기굴절과 태양 원반 크기는 무시한다.'
    },
    {
        'id': 'D1_symbolic_polar_day', 'family': 'daylength', 'perturbation': 'symbolic_regime_change',
        'prompt': COMMON + '\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용하려 한다. phi=78 deg, delta=23.44 deg이다. 이때 arccos의 인수를 먼저 계산하고, 실수 범위를 벗어날 경우 그 수학적 의미를 해석하여 천문학적 일장을 답하라. 대기굴절은 무시한다.'
    },
    {
        'id': 'D2_symbolic_polar_night', 'family': 'daylength', 'perturbation': 'symbolic_sign_change',
        'prompt': COMMON + '\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용하려 한다. phi=-78 deg, delta=23.44 deg이다. arccos의 인수를 확인하고, 실수 범위를 벗어날 경우 극야 또는 백야 중 어느 경우인지 판정하여 일장을 답하라.'
    },
]


def request_one(test: dict) -> dict:
    encoded = urllib.parse.quote(test['prompt'], safe='')
    query = urllib.parse.urlencode({
        'model': MODEL, 'seed': SEED, 'private': 'true', 'temperature': '0.2'
    })
    url = f'{BASE_URL}/{encoded}?{query}'
    started = dt.datetime.now(dt.timezone.utc)
    status = 0
    body = ''
    headers = {}
    error = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'AI-math-benchmark-fast/1.0'})
            with urllib.request.urlopen(req, timeout=120) as response:
                status = response.status
                body = response.read().decode('utf-8', errors='replace')
                headers = {k.lower(): v for k, v in response.headers.items()}
            break
        except Exception as exc:  # noqa: BLE001
            error = repr(exc)
            time.sleep(4 * (attempt + 1))
    ended = dt.datetime.now(dt.timezone.utc)
    return {
        'test_id': test['id'], 'family': test['family'], 'perturbation': test['perturbation'],
        'model_requested': MODEL, 'seed': SEED, 'temperature_requested': 0.2,
        'prompt': test['prompt'],
        'prompt_sha256': hashlib.sha256(test['prompt'].encode('utf-8')).hexdigest(),
        'started_at_utc': started.isoformat(), 'ended_at_utc': ended.isoformat(),
        'elapsed_seconds': (ended-started).total_seconds(), 'http_status': status,
        'response_headers': headers, 'raw_response': body,
        'raw_response_sha256': hashlib.sha256(body.encode('utf-8')).hexdigest(),
        'error': error,
    }

records = []
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(request_one, test) for test in TESTS]
    for future in concurrent.futures.as_completed(futures):
        record = future.result()
        records.append(record)
        print(record['test_id'], record['http_status'], round(record['elapsed_seconds'], 2), flush=True)
records.sort(key=lambda r: r['test_id'])

meta = {
    'benchmark_name': 'Math for Ocean Scientists - fast GSM-style pilot',
    'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'endpoint': BASE_URL, 'model_requested': MODEL, 'seed': SEED,
    'test_count': len(TESTS), 'response_count': len(records), 'max_parallel': 4,
    'notes': [
        'External anonymous endpoint; no user account history or personalization was supplied.',
        'The experiment is a focused stress test of one externally hosted reasoning model, not a leaderboard or a claim about all frontier systems.',
    ],
}
(OUT/'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT/'raw_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
with (OUT/'raw_records.jsonl').open('w', encoding='utf-8') as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

md = ['# External-model raw evidence', '', f'- Model: `{MODEL}`', f'- Seed: `{SEED}`', f'- UTC: `{meta["created_at_utc"]}`', '']
for r in records:
    md += [f'## {r["test_id"]}', '', '### Prompt', '', '```text', r['prompt'], '```', '', f'Prompt SHA-256: `{r["prompt_sha256"]}`', '', '### Raw response', '', '```text', r['raw_response'], '```', '', f'Response SHA-256: `{r["raw_response_sha256"]}`', '']
(OUT/'raw_evidence.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
