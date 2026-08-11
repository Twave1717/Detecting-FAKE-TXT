from __future__ import annotations

import datetime as dt
import hashlib
import json
import pathlib
import time
import urllib.error
import urllib.parse
import urllib.request

MODEL = "openai-fast"
SEEDS = [42]
BASE_URL = "https://text.pollinations.ai"
OUT = pathlib.Path("evidence")
OUT.mkdir(parents=True, exist_ok=True)

COMMON = (
    "다음 수학 문제를 단계별로 풀어라. 문제에서 요구한 해의 집합과 적용 조건을 빠뜨리지 말고, "
    "최종 답은 <final>...</final> 안에 적어라. 수치값은 가능한 경우 소수점 넷째 자리까지 제시하라."
)

TESTS = [
    {
        "id": "W0_base_all_roots",
        "family": "wave_dispersion",
        "perturbation": "base",
        "prompt": COMMON + "\n\n수심 h=20 m인 균일 수심 해역에서 관측 각주파수 omega=1 rad/s인 표면중력파가 파 진행방향의 균일 해류 U=1 m/s 위를 진행한다. g=9.81 m/s^2이고 파수 k>0은 (omega-U*k)^2 = g*k*tanh(k*h)를 만족한다. 구간 0<k<20에서 모든 양의 실근을 구하고, 각 근에 대해 고유주파수 sigma=omega-U*k의 부호를 적어라."
    },
    {
        "id": "W1_target_positive_branch",
        "family": "wave_dispersion",
        "perturbation": "question_target",
        "prompt": COMMON + "\n\n수심 h=20 m, omega=1 rad/s, U=1 m/s, g=9.81 m/s^2이다. 구간 0<k<20에서 제곱식 (omega-U*k)^2 = g*k*tanh(k*h)의 근 가운데, 제곱하기 전 양의 분기 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는 k만 구하라. 제외되는 제곱식의 근이 있다면 그 값과 제외 이유도 적어라."
    },
    {
        "id": "W2_irrelevant_information",
        "family": "wave_dispersion",
        "perturbation": "irrelevant_information",
        "prompt": COMMON + "\n\n수심 h=20 m, omega=1 rad/s, 파 진행방향 해류 U=1 m/s, g=9.81 m/s^2이다. 현장에서는 파고 0.60 m, 수온 18.2 degC, 염분 34.7, 해수밀도 1025 kg/m^3도 함께 기록했다. 구간 0<k<20에서 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는 파수 k만 구하고, 사용하지 않은 정보가 무엇인지 밝혀라."
    },
    {
        "id": "W3_symbolic_current_reversal",
        "family": "wave_dispersion",
        "perturbation": "symbolic_sign_change",
        "prompt": COMMON + "\n\n수심 h=20 m, omega=1 rad/s, g=9.81 m/s^2이다. 이번에는 해류가 파 진행방향과 반대여서 U=-1 m/s로 둔다. 구간 0<k<10에서 제곱식 (omega-U*k)^2 = g*k*tanh(k*h)의 모든 양의 실근을 구하고, 각 근이 양의 분기 omega-U*k = +sqrt(g*k*tanh(k*h))를 만족하는지도 판정하라."
    },
    {
        "id": "W4_symbolic_weak_opposing_current",
        "family": "wave_dispersion",
        "perturbation": "symbolic_value_change",
        "prompt": COMMON + "\n\n수심 h=20 m, omega=1 rad/s, g=9.81 m/s^2, U=-0.2 m/s이다. 구간 0<k<100에서 (omega-U*k)^2 = g*k*tanh(k*h)의 모든 양의 실근을 구하라. 근의 개수와 각 근에서 sigma=omega-U*k의 부호를 함께 적어라."
    },
    {
        "id": "P0_base_monotone_transform",
        "family": "probability_transform",
        "perturbation": "base",
        "prompt": COMMON + "\n\n단면에 수직인 유속 U가 구간 [0,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 구하고, 적분값이 1인지 확인하라."
    },
    {
        "id": "P1_symbolic_support_crosses_zero",
        "family": "probability_transform",
        "perturbation": "symbolic_interval_change",
        "prompt": COMMON + "\n\n단면에 수직인 유속 U가 구간 [-1,2]에서 균일분포를 따른다고 하자. Y=U^2일 때 Y의 확률밀도함수 f_Y(y)와 지지집합을 정확히 구하라. y의 범위에 따라 역상의 개수가 달라질 수 있음을 고려하고, 밀도 적분이 1인지 확인하라."
    },
    {
        "id": "P2_irrelevant_information",
        "family": "probability_transform",
        "perturbation": "irrelevant_information",
        "prompt": COMMON + "\n\n37개 정점에서 얻은 자료를 단순화하여 단면 유속 U가 [-1,2]에서 균일분포를 따른다고 가정한다. 평균 수온은 17.8 degC, 평균 염분은 34.6, 수심은 50 m였다. Y=U^2일 때 Y의 확률밀도함수와 지지집합을 구하고, 이 계산에 사용되지 않는 정보를 적어라."
    },
    {
        "id": "P3_operation_change_absolute_value",
        "family": "probability_transform",
        "perturbation": "operation_change",
        "prompt": COMMON + "\n\n유속 U가 [-1,2]에서 균일분포를 따른다. 이번에는 Y=abs(U)로 정의한다. Y의 확률밀도함수 f_Y(y)와 지지집합을 구하라. y의 구간에 따라 U의 역상이 몇 개인지도 설명하라."
    },
    {
        "id": "P4_question_target_probability_moment",
        "family": "probability_transform",
        "perturbation": "question_target",
        "prompt": COMMON + "\n\n유속 U가 [-1,2]에서 균일분포를 따르고 Y=U^2이다. 확률 P(Y<=1)과 기대값 E[Y]를 각각 정확한 분수 또는 정수로 구하라. 확률밀도함수 전체를 다시 유도할 필요는 없다."
    },
    {
        "id": "P5_misleading_claim",
        "family": "probability_transform",
        "perturbation": "misleading_statement",
        "prompt": COMMON + "\n\n유속 U가 [-1,2]에서 균일분포를 따르고 Y=U^2이다. 한 연구자는 '제곱함수는 증가함수이므로 U=sqrt(Y) 하나만 역함수로 쓰면 된다'고 주장한다. 이 주장을 검토하고, 틀렸다면 정확한 f_Y(y)와 지지집합을 제시하라."
    },
    {
        "id": "D0_base_midlatitude",
        "family": "daylength",
        "perturbation": "base",
        "prompt": COMMON + "\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용한다. phi는 위도(rad), delta는 태양 적위(rad)이다. phi=35.18 deg, delta=23.44 deg일 때 D를 시간 단위로 계산하라. 대기굴절과 태양 원반 크기는 무시한다."
    },
    {
        "id": "D1_symbolic_polar_day",
        "family": "daylength",
        "perturbation": "symbolic_regime_change",
        "prompt": COMMON + "\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용하려 한다. phi=78 deg, delta=23.44 deg이다. 이때 arccos의 인수를 먼저 계산하고, 실수 범위를 벗어날 경우 그 수학적 의미를 해석하여 천문학적 일장을 답하라. 대기굴절은 무시한다."
    },
    {
        "id": "D2_symbolic_polar_night",
        "family": "daylength",
        "perturbation": "symbolic_sign_change",
        "prompt": COMMON + "\n\n이상적인 천문학적 일장 근사식 D=(24/pi)*acos(-tan(phi)*tan(delta))를 사용하려 한다. phi=-78 deg, delta=23.44 deg이다. arccos의 인수를 확인하고, 실수 범위를 벗어날 경우 극야 또는 백야 중 어느 경우인지 판정하여 일장을 답하라."
    },
    {
        "id": "D3_irrelevant_operational_time",
        "family": "daylength",
        "perturbation": "irrelevant_information",
        "prompt": COMMON + "\n\n위도 phi=78 deg에서 태양 적위 delta=23.44 deg이다. 경도는 15 deg E이고 관측장비는 정오에 2시간 정비로 멈췄으며 기압은 1008 hPa였다. 이상적인 천문학적 일장 D=(24/pi)*acos(-tan(phi)*tan(delta))을 해석하여 하루 중 태양이 지평선 위에 있는 시간을 구하라. 대기굴절은 무시한다."
    },
]


def get_url(url: str, attempts: int = 4) -> tuple[int, str, dict[str, str]]:
    last_error = None
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "AI-math-benchmark/1.0"})
            with urllib.request.urlopen(req, timeout=240) as response:
                body = response.read().decode("utf-8", errors="replace")
                headers = {k.lower(): v for k, v in response.headers.items()}
                return response.status, body, headers
        except Exception as exc:  # noqa: BLE001
            last_error = repr(exc)
            time.sleep(5 * (attempt + 1))
    return 0, json.dumps({"error": last_error}, ensure_ascii=False), {}


model_status, model_body, model_headers = get_url(f"{BASE_URL}/models")
(OUT / "model_registry_raw.json").write_text(model_body, encoding="utf-8")

records = []
for test in TESTS:
    for seed in SEEDS:
        encoded = urllib.parse.quote(test["prompt"], safe="")
        query = urllib.parse.urlencode(
            {
                "model": MODEL,
                "seed": seed,
                "private": "true",
                "json": "true",
                "temperature": "0.2",
            }
        )
        url = f"{BASE_URL}/{encoded}?{query}"
        started = dt.datetime.now(dt.timezone.utc)
        status, body, headers = get_url(url)
        ended = dt.datetime.now(dt.timezone.utc)
        record = {
            "test_id": test["id"],
            "family": test["family"],
            "perturbation": test["perturbation"],
            "model_requested": MODEL,
            "seed": seed,
            "temperature_requested": 0.2,
            "prompt": test["prompt"],
            "prompt_sha256": hashlib.sha256(test["prompt"].encode("utf-8")).hexdigest(),
            "started_at_utc": started.isoformat(),
            "ended_at_utc": ended.isoformat(),
            "elapsed_seconds": (ended - started).total_seconds(),
            "http_status": status,
            "response_headers": headers,
            "raw_response": body,
            "raw_response_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        }
        try:
            record["parsed_response"] = json.loads(body)
        except json.JSONDecodeError:
            record["parsed_response"] = None
        records.append(record)
        time.sleep(2)

run_meta = {
    "benchmark_name": "Math for Ocean Scientists - GSM-style perturbation pilot",
    "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "endpoint": BASE_URL,
    "model_registry_http_status": model_status,
    "model_registry_headers": model_headers,
    "model_requested": MODEL,
    "seeds": SEEDS,
    "test_count": len(TESTS),
    "response_count": len(records),
    "notes": [
        "External anonymous endpoint; no user account history or personalization was supplied.",
        "The registry response is archived alongside outputs.",
        "This pilot evaluates one externally hosted open reasoning model and does not represent every commercial frontier model.",
    ],
}

(OUT / "run_metadata.json").write_text(json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8")
(OUT / "raw_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
with (OUT / "raw_records.jsonl").open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

md = [
    "# External-model raw evidence",
    "",
    f"- Model requested: `{MODEL}`",
    f"- Seeds: `{SEEDS}`",
    f"- UTC generated: `{run_meta['created_at_utc']}`",
    "- Responses below are unedited raw bodies returned by the endpoint.",
    "",
]
for r in records:
    md.extend(
        [
            f"## {r['test_id']} / seed {r['seed']}",
            "",
            "### Prompt",
            "",
            "```text",
            r["prompt"],
            "```",
            "",
            f"Prompt SHA-256: `{r['prompt_sha256']}`",
            "",
            "### Raw response",
            "",
            "```json",
            r["raw_response"],
            "```",
            "",
            f"Response SHA-256: `{r['raw_response_sha256']}`",
            "",
        ]
    )
(OUT / "raw_evidence.md").write_text("\n".join(md), encoding="utf-8")
print(json.dumps(run_meta, ensure_ascii=False, indent=2))
