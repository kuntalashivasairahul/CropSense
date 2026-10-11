---
name: guidellm-serving
description: Benchmark latency, throughput, and reliability of an LLM served through an OpenAI-compatible or vLLM endpoint with GuideLLM. Do not use it to measure classifier accuracy.
---

# GuideLLM serving benchmarks

Use [GuideLLM](https://github.com/vllm-project/guidellm) when a task has a running LLM inference endpoint and asks how it performs under realistic request traffic. First verify the current CLI and supported platforms in the upstream README, since they can change.

Keep GuideLLM in an isolated environment. Do not add it to the CropSense app requirements: this project's MobileNetV2, NDVI Random Forest, and pest LSTM are evaluated by their own Kaggle notebooks with labelled examples. GuideLLM reports serving performance, not classification accuracy.

Before a benchmark, record the endpoint, model and server versions, hardware, input dataset or synthetic request sizes, traffic profile, warmup, duration, and the latency or throughput target. Run the same fixed scenario for each candidate and save GuideLLM's JSON/CSV results with the configuration. Compare error rate and latency distribution alongside throughput; keep accuracy evaluation separate.

The upstream quick start uses `guidellm run --backend kind=openai_http,target=http://localhost:8000 --profile kind=sweep --constraint kind=max_duration,seconds=30 --data kind=synthetic_text,prompt_tokens=256,output_tokens=128`. Adapt the endpoint and workload to the actual application. Do not claim a benchmark was run unless an endpoint and results exist.
