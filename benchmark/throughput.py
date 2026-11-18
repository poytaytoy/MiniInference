"""Measure aggregate throughput with concurrent HTTP requests."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

from client import parser, positive_int, print_latency, request


def main():
    options = parser(__doc__)
    options.add_argument("--concurrency", type=positive_int, default=4)
    args = options.parse_args()

    request(args, stream=False)
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        start = perf_counter()
        futures = [executor.submit(request, args, stream=False) for _ in range(args.requests)]
        results = [future.result() for future in futures]
        elapsed = perf_counter() - start

    completions = sum(result.completion_tokens for result in results)
    print(f"{len(results)} requests in {elapsed:.2f} s at concurrency {args.concurrency}")
    print(f"{len(results) / elapsed:.2f} requests/s, {completions / elapsed:.2f} completion tokens/s")
    print_latency("Response latency", (result.seconds for result in results))


if __name__ == "__main__":
    main()
