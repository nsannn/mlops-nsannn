# scripts/load_test.py
"""Basic load test for the inference API: N POST /predict requests from C parallel clients.

Run (API reachable at --url, e.g. through kubectl port-forward -n mlops svc/emotion-api 8000:80):
  python scripts/load_test.py --n 100 --concurrency 1
Prints p50/p95/p99 latency (ms), throughput (req/s) and errors as JSON.
"""
import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TEXTS = ["i feel so happy and grateful today", "this traffic is making me really angry",
         "i am scared of what the doctor will say", "i miss you so much it hurts",
         "wow i did not expect that at all", "i love spending sundays with my family"]


def call(url: str, text: str) -> tuple[float, bool]:
    req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
            ok = resp.status == 200
    except urllib.error.URLError:  # includes HTTP 4xx/5xx
        ok = False
    return (time.perf_counter() - t0) * 1000, ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    for text in TEXTS:  # warm-up, not measured
        call(args.url, text)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.concurrency) as pool:
        results = list(pool.map(lambda i: call(args.url, TEXTS[i % len(TEXTS)]), range(args.n)))
    wall = time.perf_counter() - t0
    q = statistics.quantiles([ms for ms, _ in results], n=100)
    print(json.dumps({"requests": args.n, "concurrency": args.concurrency,
                      "errors": sum(not ok for _, ok in results),
                      "p50_ms": round(q[49], 1), "p95_ms": round(q[94], 1), "p99_ms": round(q[98], 1),
                      "throughput_rps": round(args.n / wall, 1)}, indent=2))


if __name__ == "__main__":
    main()
