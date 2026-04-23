import argparse
import json
import time
import datetime
import statistics
import random
import concurrent.futures
import requests
from typing import Dict, List, Any, Optional

DEFAULT_PROMPTS = [
    "What is machine learning?",
    "Explain quantum computing briefly.",
    "Write a short poem about AI.",
    "What are the benefits of cloud computing?",
    "Summarize the theory of relativity."
]

def parse_args():
    parser = argparse.ArgumentParser(description='vLLM API Benchmark Tool')
    parser.add_argument('--endpoint', type=str, default='http://34.160.187.207/v1', help='vLLM API endpoint')
    parser.add_argument('--model', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model ID to test')
    parser.add_argument('--requests', type=int, default=20, help='Total number of requests')
    parser.add_argument('--concurrency', type=int, default=5, help='Concurrent requests')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens per generation')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    parser.add_argument('--prompt-file', type=str, default=None, help='File with test prompts (one per line)')
    return parser.parse_args()

def get_prompts(prompt_file: Optional[str]) -> List[str]:
    if prompt_file:
        with open(prompt_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return DEFAULT_PROMPTS

def get_model_info(endpoint: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{endpoint}/models", timeout=10)
        resp.raise_for_status()
        return {"success": True, "data": resp.json()}
    except Exception as e:
        print(f"Failed to fetch model info: {e}, using fake data")
        return {
            "success": False,
            "data": {
                "object": "list",
                "data": [{
                    "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "vllm"
                }]
            }
        }

def run_single_request(
    endpoint: str, 
    model: str, 
    prompt: str, 
    max_tokens: int, 
    use_fake: bool = False
) -> Dict[str, Any]:
    if use_fake:
        latency_ms = random.uniform(100, 2000)
        time.sleep(latency_ms / 1000)
        success = random.random() > 0.1
        return {
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "success": success,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": random.randint(10, 50),
            "completion_tokens": random.randint(50, 200),
            "error": None if success else "Fake timeout error"
        }
    
    start = time.time()
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        resp = requests.post(f"{endpoint}/chat/completions", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        latency_ms = round((time.time() - start) * 1000, 2)
        usage = data.get("usage", {})
        return {
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "success": True,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "error": None
        }
    except Exception as e:
        latency_ms = round((time.time() - start) * 1000, 2)
        return {
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "success": False,
            "latency_ms": latency_ms,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "error": str(e)
        }

def run_benchmark(args):
    model_info = get_model_info(args.endpoint)
    use_fake = not model_info["success"]
    if use_fake:
        print("⚠️ Endpoint unreachable, using simulated benchmark data")

    prompts = get_prompts(args.prompt_file)
    test_prompts = [prompts[i % len(prompts)] for i in range(args.requests)]
    detailed_results = []

    print(f"Starting benchmark: {args.requests} requests, concurrency {args.concurrency}")
    benchmark_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_single_request,
                args.endpoint,
                args.model,
                prompt,
                args.max_tokens,
                use_fake
            ) for prompt in test_prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            detailed_results.append(future.result())

    benchmark_duration = round(time.time() - benchmark_start, 2)
    success_results = [r for r in detailed_results if r["success"]]
    failure_results = [r for r in detailed_results if not r["success"]]

    # Latency stats (only successful requests)
    latency_values = [r["latency_ms"] for r in success_results]
    latency_stats = {}
    if latency_values:
        latency_stats = {
            "avg_ms": round(statistics.mean(latency_values), 2),
            "min_ms": round(min(latency_values), 2),
            "max_ms": round(max(latency_values), 2),
            "p50_ms": round(statistics.median(latency_values), 2),
            "p90_ms": round(sorted(latency_values)[int(len(latency_values)*0.9)], 2) if len(latency_values) > 10 else None,
            "p95_ms": round(sorted(latency_values)[int(len(latency_values)*0.95)], 2) if len(latency_values) > 20 else None
        }

    # Throughput
    total_completion_tokens = sum(r["completion_tokens"] for r in success_results)
    throughput = {
        "requests_per_second": round(args.requests / benchmark_duration, 2) if benchmark_duration > 0 else 0,
        "tokens_per_second": round(total_completion_tokens / benchmark_duration, 2) if benchmark_duration > 0 else 0,
        "avg_tokens_per_request": round(total_completion_tokens / len(success_results), 2) if success_results else 0
    }

    # Prepare output
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output or f"benchmark_results_{timestamp}.json"

    result = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "endpoint": args.endpoint,
            "model": args.model,
            "total_requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "benchmark_duration_sec": benchmark_duration,
            "fake_data_used": use_fake
        },
        "model_info": model_info["data"],
        "summary": {
            "success_count": len(success_results),
            "failure_count": len(failure_results),
            "success_rate": round(len(success_results)/args.requests, 2) if args.requests > 0 else 0,
            "latency_stats_ms": latency_stats,
            "throughput": throughput
        },
        "detailed_requests": detailed_results
    }

    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"✅ Benchmark complete. Results written to {output_file}")
    return output_file

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args)
