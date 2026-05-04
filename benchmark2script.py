#!/usr/bin/env python3
"""benchmark2script.py - Generate vLLM start scripts from LocalMaxxing benchmark URLs.

Takes a LocalMaxxing benchmark run URL, fetches the benchmark parameters from the API,
and generates a new shell script that codifies those parameters for repeated use.
"""

import sys
import re
import json
import shutil
import argparse
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Template script to use as base
TEMPLATE_SCRIPT = Path(__file__).parent / "start-vllm.sh"
API_BASE = "https://www.localmaxxing.com/api/benchmarks"


def fetch_benchmark(run_id: str) -> dict:
    """Fetch benchmark data from the LocalMaxxing API for a specific run."""
    import urllib.request
    
    # The API doesn't support filtering by run id, so we fetch all and find the match
    url = f"{API_BASE}"
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "benchmark2script/1.0")
    
    with urllib.request.urlopen(req, timeout=30) as response:
        data = json.loads(response.read().decode())
    
    benchmarks = data.get("benchmarks", [])
    
    # Find the matching benchmark by run id
    for benchmark in benchmarks:
        if benchmark["id"] == run_id:
            return benchmark
    
    # If not found, show what's available
    print("ERROR: Benchmark run not found in the Public benchmarks database", file=sys.stderr)
    print(f"  Searched for: {run_id}", file=sys.stderr)
    print(f"  Available vllm benchmarks:", file=sys.stderr)
    for b in benchmarks:
        if b["engine"]["engineName"] == "vllm":
            model = b["model"]["hfId"]
            print(f"    {b['id'][:20]}... | {model[:30]} | tokSOut={b.get('tokSOut', '?')}", file=sys.stderr)
    sys.exit(1)


def extract_params(benchmark: dict) -> dict:
    """Extract vLLM start parameters from the benchmark data."""
    flags = benchmark.get("engineFlags", {})
    engine = benchmark.get("engine", {})
    model = benchmark.get("model", {})
    hardware = benchmark.get("hardware", {})
    
    params = {
        # Model info
        "model": flags.get("model", model.get("hfId", "")),
        "model_revision": benchmark.get("modelRevision", "main"),
        
        # Engine config  
        "tensor_parallel": flags.get("tensorParallel"),
        "pipeline_parallel": flags.get("pipelineParallel"),
        "gpu_memory_utilization": flags.get("gpuMemUtil"),
        "gpu_cache_size_mb": flags.get("kvCacheSizeMb"),
        "quantization": flags.get("engineQuant") or flags.get("sglangQuant") or engine.get("quantization", ""),
        
        # Inference config
        "max_model_len": flags.get("maxModelLen") or benchmark.get("contextLength"),
        "max_num_seqs": flags.get("maxRunningSeqs", flags.get("concurrency")),
        "max_num_batched_tokens": flags.get("maxNumBatchedTokens"),
        "dtype": flags.get("dtype"),
        "context_length": flags.get("contextLength") or benchmark.get("contextLength"),
        
        # Optimization flags
        "flash_attn": flags.get("flashAttn") or False,
        "chunked_prefill": flags.get("chunkedPrefill") or False,
        "spec_decoding": flags.get("specDecoding") or False,
        "spec_method": flags.get("specMethod"),
        "mtp_enabled": flags.get("mtpEnabled") or False,
        "mtp_draft_layers": flags.get("mtpDraftLayers"),
        
        # Communication
        "distributed_backend": flags.get("distributedBackend"),
        
        # Quantization
        "quantization": flags.get("quantization") or engine.get("quantization", ""),
        
        # extra flags from command
        "extra_flags": parse_extra_flags(flags.get("extraFlags", "")),
    }
    
    # Extract hardware info for reference
    if hardware:
        params["hardware"] = {
            "hw_class": hardware.get("hwClass"),
            "gpu_name": hardware.get("gpuName"),
            "gpu_count": hardware.get("gpuCount"),
            "vram_gb": hardware.get("vramGb"),
            "unified_memory_gb": hardware.get("unifiedMemoryGb"),
            "cpu": hardware.get("cpu"),
            "os": hardware.get("os"),
        }
    
    return params


def parse_extra_flags(extra_flags: str) -> dict:
    """Parse the extraFlags field which contains vLLM CLI arguments."""
    parsed = {}
    if not extra_flags:
        return parsed
    
    # Extract --key values from the flags string
    for match in re.finditer(r'--([a-z0-9_-]+)\s+([^\s]+)', extra_flags):
        flag_name = match.group(1)
        flag_value = match.group(2)
        
        # Skip environment variables like ONEAPI_DEVICE_SELECTOR=
        if "=" in flag_value:
            key, val = flag_value.split("=", 1)
            parsed[f"env_{key}"] = val
        else:
            # Try to parse boolean flags
            parsed[flag_name] = flag_value
    
    # Check for environment variables at the start
    env_pattern = r'([A-Z_]+)=([^\s]+)'
    for match in re.finditer(env_pattern, extra_flags):
        if not match.group(1).startswith('_'):
            parsed[f"env_{match.group(1)}"] = match.group(2)
    
    return parsed


def generate_script(template_path: Path, benchmark: dict, script_name: str) -> Path:
    """Generate a new start-vllm script based on the benchmark parameters."""
    params = extract_params(benchmark)
    
    # Read template
    template = template_path.read_text()
    
    # Generate the new script
    output_path = Path.cwd() / f"start-vllm-{script_name}.sh"
    
    # Add benchmark metadata to the top of the script
    metadata = f"""
# Generated from LocalMaxxing benchmark:
#   URL: https://www.localmaxxing.com/models/{benchmark['model']['hfId']}?engineName=vllm&run={benchmark['id']}
#   Benchmark ID: {benchmark['id']}
#   Created: {params.get('created', 'just now')}
"""
    
    # Build the vLLM serve command from flags
    env_vars = []
    vllm_args = ["\\n    "]  # Start with indentation
    
    for key, value in params.items():
        if key.startswith("env_"):
            env_var = key[4:]
            env_vars.append(f"export {env_var}={value!r}")
        elif key in ["tensor_parallel"]:
            vllm_args.append(f"--tensor-parallel-size {value}")
        elif key in ["pipeline_parallel"]:
            vllm_args.append(f"--pipeline-parallel-size {value}")
        elif key in ["gpu_memory_utilization"]:
            vllm_args.append(f"--gpu-memory-utilization {value}")
        elif key in ["gpu_cache_size_mb"]:
            vllm_args.append(f"--gpu-cache-size-mb {value}")
        elif key in ["quantization"]:
            vllm_args.append(f"--quantization {value}")
        elif key in ["max_model_len", "context_length"]:
            vllm_args.append(f"--max-model-len {value}")
        elif key in ["max_num_seqs", "max_running_seqs", "max_num_batched_tokens"]:
            vllm_args.append(f"--max-num-batched-tokens {value}")
        elif key in ["max_num_seqs"]:
            vllm_args.append(f"--max-num-seqs {value}")
        elif key in ["dtype"]:
            vllm_args.append(f"--dtype {value}")
        elif key == "flash_attn" and value:
            vllm_args.append("--flash-attn")
        elif key == "chunked_prefill" and value:
            vllm_args.append("--enable-chunked-prefill")
        elif key in ["spec_decoding"]:
            vllm_args.append(f"--speculative-decoding-method {value}")
        elif key in ["spec_method"]:
            vllm_args.append(f"--speculative-method {value}")
        elif key == "mtp_enabled" and value:
            vllm_args.append("--mtp-enabled")
        elif key in ["mtp_draft_layers"]:
            vllm_args.append(f"--spec-num-draft-tokens-per-step {value}")
        elif key in ["distributed_backend"]:
            vllm_args.append(f"--distributed-executor-backend {value}")
        elif key == "scheduler_delay_factor":
            vllm_args.append(f"--scheduler-delay-factor {value}")
        elif key == "disable_custom_all_reduce":
            vllm_args.append("--disable-custom-all-reduce")
        elif key == "concurrency":
            vllm_args.append(f"--num-scheduler-workers {value}")
        elif key == "attention_backend":
            vllm_args.append(f"--attention-backend {value}")
        elif key == "scheduler_delay_factor":
            vllm_args.append(f"--scheduler-delay-factor {value}")
        else:
            # Unhandled flag - write as comment for the user to review
            env_vars.append(f"# TODO: {key}={value!r}")
    
    # Clean up vllm_args - remove any duplicates like \\n    from earlier
    vllm_args = [arg for arg in vllm_args if not (isinstance(arg, str) and arg.strip() == "")]
    
    # Write the new script
    with open(output_path, "w") as f:
        f.write(template[:template.find("# ---- launch -----")])
        
        # Add vLLM-specific section
        f.write(f"# ---- configuration generated from benchmark ------\n")
        f.write(f"# Model: {params['model']}\n")
        f.write(f"# Quantization: {params['quantization']}\n")
        f.write(f"# Tensor Parallel: {params.get('tensor_parallel')}\n")
        f.write(f"# Context Length: {params.get('context_length') or params.get('max_model_len')}\n")
        f.write(f"#\n")
        f.write(f"# Generated from: https://www.localmaxxing.com/models/{benchmark['model']['hfId']}?engineName=vllm&run={benchmark['id']}\n")
        f.write(f"#\n")
        f.write(f"# ---- start vllm ------ ------ ------ ------ ------ ------ ---\n")
        f.write(f"\n")
        
        # Add environment variables from extraFlags
        if env_vars:
            f.write(f"\n")
            for env in env_vars:
                f.write(f"{env}\n")
            f.write(f"\n")
        
        # Add the vLLM start command
        model_path = f'"/home/michel/models/{params["model"]}"'
        max_tokens = params.get("context_length") or params.get("max_model_len")
        
        f.write(f"nohup $VLLM_VENV/bin/vllm serve \\\n")
        f.write(f"  {model_path} \\\n")
        f.write(f"  --host \"$HOST\" \\\n")
        f.write(f"  --port \"$PORT\" \\\n")
        f.write(f"  --dtype \"$VLLM_DTYPE:-auto\" \\\n")
        f.write(f"  --quantization \"$VLLM_QUANT:{params['quantization']}\" \\\n")
        f.write(f"  --max-model-len \"$MAX_MODEL_LEN:{max_tokens}\" \\\n")
        f.write(f"  --gpu-memory-utilization \"$VLLM_GPU_MEM:0.95\" \\\n")
        f.write(f"  --tensor-parallel-size \"$VLLM_TP:{params.get('tensor_parallel','2')}\" \\\n")
        f.write(f"  --max-num-batched-tokens \"$VLLM_MAX_SEQS:{params.get('context_length') or params.get('max_model_len')}\" \\\n")
        
        # Add other common args from the benchmark
        if params.get("engineFlags", {}).get("disable_custom_all_reduce"):
            f.write(f"  --disable-custom-all-reduce \\\n")
        if params.get("engineFlags", {}).get("max_num_seqs"):
            f.write(f"  --max-num-seqs \"$VLLM_MAX_SEQS:{params['engineFlags'].get('max_num_seqs')}\" \\\n")
        if params.get("engineFlags", {}).get("scheduler_delay_factor"):
            f.write(f"  --scheduler-delay-factor \"{params['engineFlags'].get('scheduler_delay_factor')}\" \\\n")
        if params.get("engineFlags", {}).get("context_length"):
            f.write(f"  --context-length {params['engineFlags'].get('context_length')} \\\n")
        
        f.write(f"  2>&1 &>> /tmp/vllm-{script_name}.log &\n")
        
        f.write(f"\n")
        f.write(f"VLLM_PID=$!\n")
        f.write(f'echo "{script_name} started (pid $VLLM_PID) on port $PORT"\n')
        f.write(f'echo "Monitor with: tail -f /tmp/vllm-{script_name}.log"\n')
        f.write(f'echo "Test with: curl http://127.0.0.1:$PORT/v1/models"\n')
        
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate vLLM start scripts from LocalMaxxing benchmark URLs"
    )
    parser.add_argument(
        "url",
        help="LocalMaxxing benchmark URL (e.g. https://www.localmaxxing.com/models/Qwen/Qwen3.6-27B?engineName=vllm&run=cmof8rjey000hl804orjkwuhd)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output script name (default: auto-generated from benchmark ID)",
        dest="output_name"
    )
    parser.add_argument(
        "--template", "-t",
        help="Path to template script (default: start-vllm.sh in same directory)",
        default=TEMPLATE_SCRIPT
    )
    
    args = parser.parse_args()
    
    # Parse the run ID from the URL
    parsed = urlparse(args.url)
    params = parse_qs(parsed.query)
    
    run_ids = params.get("run", [])
    if not run_ids:
        print(f"Error: No 'run' parameter in URL", file=sys.stderr)
        print(f"URL: {args.url}", file=sys.stderr)
        sys.exit(1)
    
    run_id = run_ids[0]
    
    # Determine output name
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = run_id[:12]  # Use first 12 chars of run ID
    
    print(f"Fetching benchmark {run_id} from LocalMaxxing...")
    benchmark = fetch_benchmark(run_id)
    
    print(f"Generating start script: {output_name}")
    output = generate_script(Path(args.template), benchmark, output_name)
    print(f"Script written to: {output}");
    print(f"\nTo run: bash {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
