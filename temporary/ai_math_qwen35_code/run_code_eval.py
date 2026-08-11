from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import re
import subprocess
import urllib.request

MODEL = 'qwen3.5:4b'
SEED = 42
BASE_URL = 'http://127.0.0.1:11434'
OUT = pathlib.Path('evidence_qwen35_code')
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    {
        'id': 'C0_general_root_code',
        'condition': 'general_implementation_request',
        'prompt': '''아래 사양을 만족하는 base R 코드만 출력하라. 설명이나 마크다운은 출력하지 않는다.

함수명: find_wave_roots
함수 인자: U, h=20, omega=1, g=9.81, kmin=1e-6, kmax=20, tol=1e-8
목적: F(k)=(omega-U*k)^2-g*k*tanh(k*h)의 구간 kmin<k<kmax 안에 있는 모든 양의 실근을 찾아 오름차순 numeric vector로 반환한다. 같은 근을 중복 반환하지 않는다. 외부 패키지는 사용하지 않는다.'''
    },
    {
        'id': 'C1_invariant_aware_root_code',
        'condition': 'mathematically_constrained_request',
        'prompt': '''아래 사양을 만족하는 base R 코드만 출력하라. 설명이나 마크다운은 출력하지 않는다.

함수명: find_wave_roots
함수 인자: U, h=20, omega=1, g=9.81, kmin=1e-6, kmax=20, tol=1e-8
목적: F(k)=(omega-U*k)^2-g*k*tanh(k*h)의 구간 kmin<k<kmax 안에 있는 모든 양의 실근을 찾아 오름차순 numeric vector로 반환한다. 같은 근을 중복 반환하지 않는다. 외부 패키지는 사용하지 않는다.

검증 조건:
1. 격자에서 F의 부호가 바뀌는 구간만 uniroot로 찾는 방식은 접하면서 부호가 바뀌지 않는 짝수 중근을 놓칠 수 있다.
2. 따라서 부호변화 구간뿐 아니라 F'(k)=0인 정지점 또는 |F|의 국소최소도 검사하여 중근을 포함해야 한다.
3. 반환 전 각 후보에서 abs(F(k))가 허용오차를 만족하는지 검사한다.
4. 다음 경계상황을 처리한다: 근 2개, 서로 가까운 근 2개, 접하는 중근 1개, 근 없음.'''
    },
]

TEST_CASES = [
    {'name': 'following_current_two_roots', 'U': 1.0, 'kmax': 20.0, 'expected': [0.0893947950686009, 11.724710044323515]},
    {'name': 'opposing_current_two_roots', 'U': -1.0, 'kmax': 10.0, 'expected': [0.131953038823522, 7.679788073521478]},
    {'name': 'near_blocking_two_close_roots', 'U': -2.44, 'kmax': 2.0, 'expected': [0.35522397396549815, 0.4728530711516974]},
    {'name': 'critical_tangent_double_root', 'U': -2.452499595121764, 'kmax': 2.0, 'expected': [0.4077494597470591]},
    {'name': 'beyond_blocking_no_root', 'U': -2.46, 'kmax': 2.0, 'expected': []},
]


def post_chat(prompt: str) -> tuple[int, str, dict[str, str]]:
    payload = {
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': True,
        'options': {
            'seed': SEED,
            'temperature': 1.0,
            'top_p': 0.95,
            'top_k': 20,
            'presence_penalty': 1.5,
            'num_predict': 2200,
            'num_ctx': 8192,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = urllib.request.Request(
        BASE_URL + '/api/chat', data=data,
        headers={'Content-Type': 'application/json', 'User-Agent': 'ocean-math-code-benchmark/1.0'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=1500) as response:
        body = response.read().decode('utf-8', errors='replace')
        return response.status, body, {k.lower(): v for k, v in response.headers.items()}


def extract_r_code(content: str) -> str:
    fenced = re.findall(r'```(?:r|R)?\s*\n(.*?)```', content, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip() + '\n'
    text = content.strip()
    text = re.sub(r'^<final>\s*', '', text)
    text = re.sub(r'\s*</final>$', '', text)
    return text.strip() + '\n'


def run_r_evaluator(code_path: pathlib.Path, result_path: pathlib.Path) -> dict:
    test_json = json.dumps(TEST_CASES, ensure_ascii=False)
    evaluator = f'''options(warn=2)
source({json.dumps(str(code_path))})
if (!exists("find_wave_roots", mode="function")) stop("find_wave_roots function not found")
F_ref <- function(k, U, h=20, omega=1, g=9.81) (omega-U*k)^2-g*k*tanh(k*h)
dedup <- function(x, tol=1e-4) {{
  x <- sort(as.numeric(x[is.finite(x)]))
  if(length(x) <= 1) return(x)
  keep <- c(TRUE, diff(x) > tol)
  x[keep]
}}
match_roots <- function(actual, expected, root_tol=2e-3) {{
  actual <- dedup(actual)
  expected <- sort(as.numeric(expected))
  if(length(actual) != length(expected)) return(FALSE)
  if(length(expected) == 0) return(TRUE)
  all(abs(actual-expected) <= root_tol)
}}
tests <- jsonlite::fromJSON({json.dumps(test_json)})
results <- list()
for(i in seq_len(nrow(tests))) {{
  tc <- tests[i,]
  err <- NULL
  actual <- NULL
  tryCatch({{
    actual <- find_wave_roots(U=tc$U, h=20, omega=1, g=9.81, kmin=1e-6, kmax=tc$kmax, tol=1e-8)
    if(!is.numeric(actual)) stop("return value is not numeric")
  }}, error=function(e) err <<- conditionMessage(e))
  if(is.null(actual)) actual <- numeric(0)
  actual <- dedup(actual)
  residuals <- if(length(actual)) abs(F_ref(actual, tc$U)) else numeric(0)
  expected <- unlist(tc$expected)
  pass_roots <- is.null(err) && match_roots(actual, expected)
  pass_residual <- is.null(err) && (length(residuals)==0 || all(residuals <= 1e-5))
  results[[i]] <- list(name=tc$name, U=tc$U, expected=as.numeric(expected), actual=as.numeric(actual), residuals=as.numeric(residuals), error=err, pass_roots=pass_roots, pass_residual=pass_residual, pass=pass_roots && pass_residual)
}}
summary <- list(test_count=length(results), passed=sum(vapply(results, function(x) isTRUE(x$pass), logical(1))), all_pass=all(vapply(results, function(x) isTRUE(x$pass), logical(1))), results=results)
jsonlite::write_json(summary, {json.dumps(str(result_path))}, auto_unbox=TRUE, pretty=TRUE, null="null")
'''
    eval_path = result_path.with_suffix('.R')
    eval_path.write_text(evaluator, encoding='utf-8')
    proc = subprocess.run(['Rscript', str(eval_path)], capture_output=True, text=True, timeout=600)
    payload = {'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    if result_path.exists():
        payload['evaluation'] = json.loads(result_path.read_text(encoding='utf-8'))
    else:
        payload['evaluation'] = None
    return payload

records = []
for item in PROMPTS:
    started = dt.datetime.now(dt.timezone.utc)
    status = 0
    raw = ''
    headers = {}
    error = None
    try:
        status, raw, headers = post_chat(item['prompt'])
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    ended = dt.datetime.now(dt.timezone.utc)
    parsed = None
    content = ''
    thinking = ''
    try:
        parsed = json.loads(raw) if raw else None
        message = (parsed or {}).get('message') or {}
        content = message.get('content', '')
        thinking = message.get('thinking', '')
    except json.JSONDecodeError:
        pass
    code = extract_r_code(content)
    code_path = OUT / f'{item["id"]}_generated.R'
    code_path.write_text(code, encoding='utf-8')
    eval_result_path = OUT / f'{item["id"]}_evaluation.json'
    try:
        execution = run_r_evaluator(code_path, eval_result_path)
    except Exception as exc:  # noqa: BLE001
        execution = {'returncode': None, 'stdout': '', 'stderr': repr(exc), 'evaluation': None}
    record = {
        'test_id': item['id'],
        'condition': item['condition'],
        'model_requested': MODEL,
        'seed': SEED,
        'prompt': item['prompt'],
        'prompt_sha256': hashlib.sha256(item['prompt'].encode('utf-8')).hexdigest(),
        'started_at_utc': started.isoformat(),
        'ended_at_utc': ended.isoformat(),
        'elapsed_seconds': (ended-started).total_seconds(),
        'http_status': status,
        'response_headers': headers,
        'raw_response': raw,
        'raw_response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        'thinking': thinking,
        'answer_content': content,
        'extracted_r_code': code,
        'model_error': error,
        'execution': execution,
    }
    records.append(record)
    print(item['id'], status, (execution.get('evaluation') or {}).get('passed'), flush=True)

meta = {
    'benchmark_name': 'Ocean wave dispersion code-generation robustness test',
    'created_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'model_requested': MODEL,
    'seed': SEED,
    'prompts': len(PROMPTS),
    'test_cases': TEST_CASES,
    'objective': 'Compare an unconstrained implementation request with a mathematically constrained request using the same executable test suite, including a tangent double root that does not change sign.',
    'notes': [
        'Fresh CI process; no user-account memory or personalization.',
        'Generated R code is executed, not judged by another language model.',
        'Passing requires correct root count, numeric accuracy, and small residual for all five cases.',
    ],
}
(OUT / 'run_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
(OUT / 'raw_and_execution_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')

md = ['# Qwen3.5 executable R-code evidence', '', f'- Model: `{MODEL}`', f'- Seed: `{SEED}`', f'- Generated UTC: `{meta["created_at_utc"]}`', '']
for record in records:
    md.extend([
        f'## {record["test_id"]}', '', '### Prompt', '', '```text', record['prompt'], '```', '',
        f'Prompt SHA-256: `{record["prompt_sha256"]}`', '', '### Raw model answer', '', '```text', record['answer_content'], '```', '',
        '### Extracted R code', '', '```r', record['extracted_r_code'], '```', '',
        '### Execution result', '', '```json', json.dumps(record['execution'], ensure_ascii=False, indent=2), '```', '',
        f'Response SHA-256: `{record["raw_response_sha256"]}`', ''
    ])
(OUT / 'raw_evidence.md').write_text('\n'.join(md), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False, indent=2))
