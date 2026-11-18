"""Measure sequential response latency over the server's HTTP API."""

from client import parser, print_latency, request


def main():
    options = parser(__doc__)
    options.add_argument("--warmup", type=int, default=1,
                         help="number of unmeasured requests")
    options.add_argument("--non-streaming", action="store_true",
                         help="measure complete JSON responses")
    args = options.parse_args()
    if args.warmup < 0:
        options.error("--warmup must be zero or greater")
    stream = not args.non_streaming

    for _ in range(args.warmup):
        request(args, stream=stream)

    results = []
    for index in range(1, args.requests + 1):
        result = request(args, stream=stream)
        results.append(result)
        first = result.first_text_seconds
        first_text = "" if not stream else (
            ", first text unavailable" if first is None else f", first text {first * 1000:.1f} ms"
        )
        print(f"Request {index}: {result.seconds * 1000:.1f} ms{first_text}, "
              f"{result.prompt_tokens} prompt / {result.completion_tokens} completion tokens")

    print_latency("Response latency", (result.seconds for result in results))
    if stream:
        print_latency("Time to first text", (
            result.first_text_seconds for result in results if result.first_text_seconds is not None
        ))


if __name__ == "__main__":
    main()
