# EchoCoder

Artifact for Let LLMs Assume: From Implicit Code Commitments to Explicit Requirement Clarification through Back-Translation.

## Setup

Use Python 3.11.16

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[hilbench]'
python -m pip install -e benchmarks/ClarifyCodeBench
```

HiL-Bench SWE also needs Docker and its separate environment:

```bash
cd benchmarks/hil-bench
uv sync
cd ../..
docker info
```

See [the HiL-Bench setup guide](benchmarks/hil-bench/README.md) for its external task images. ClarifyCodeBench functional evaluation needs the corresponding LiveCodeBench executable tests.


```bash
export LLM_API_KEY='<your-key>'
export LLM_BASE_URL='<your-endpoint>'
```

## Models and data

The paper uses **GPT-5.6-Luna** and **Claude Haiku 4.5**, an 8,192-output-token limit. 

- `benchmarks/ClarifyCodeBench/data/ClarifyCodeBench.jsonl`: 419 standalone programming tasks with clarification annotations.
- `benchmarks/hil-bench/harbor_swe/`: 100 public software-engineering tasks with 353 annotated blockers. `benchmarks/hil-bench-data/` contains associated metadata. 

We sets `K_r=2` reconstructions and `K_q=3` question-generation calls per reconstruction, for at most six question-generation calls. The script defaults use `3 × 2`, so set the paper values explicitly:

```bash
export ECHOCODER_MODEL=gpt-5.6-luna
export ECHOCODER_KR=2
export ECHOCODER_KQ=3
```

The Default, GPT Engineer, ClarifyGPT, and code-ablation runners also use a six-call budget.

## Outputs

EchoCoder and the ClarifyCodeBench runners write `results.jsonl`, per-task logs, and token/cost records in method-specific `output/` directories. Question-asking methods write `summary.question.json`, and evaluated methods write `summary.passrate.json`; the original HiL-Bench runner uses its own `public_metrics/` format. The main locations are:

```text
echocoder/output/clarifycodebench/<model>/echocoder/
echocoder/output/hilbench_swe/<model>/echocoder_sweagent_progressive/
baselines/<method>/output/clarifycodebench/<model>/<method>/
benchmarks/hil-bench/results/original_ask_human/<model>/
```

The paper reports means over five runs; one command below produces one run. Override each method's `*_OUTPUT_DIR` variable to keep repeats separate. 

## Run the methods

Each question-asking ClarifyCodeBench command obtains oracle answers, generates final code, evaluates it, and writes both question and Pass@1 summaries in one run. Full Information generates and evaluates code from the complete requirements. Provide the same external LiveCodeBench evaluator to every method. Its command may use `{code_file}`, `{task_id}`, `{question_id}`, and `{sample_idx}`; exit status 0 means the code passes.

```bash
export CLARIFY_PASS_COMMAND='<evaluator command using {code_file} and {question_id}>'
bash scripts/run_echocoder_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
bash scripts/run_vanilla_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
bash scripts/run_gptengineer_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
bash scripts/run_clarifygpt_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
bash scripts/run_fullinfo_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
bash scripts/run_ablation_reconstruction_clarifycodebench.sh --pass-command "$CLARIFY_PASS_COMMAND"
```

Run the HiL-Bench SWE methods with their Docker images and benchmark runtime available:

```bash
bash scripts/run_echocoder_hilbench_sweagent.sh --progressive-edit
HILBENCH_PASSES=1 bash scripts/run_hilbench_original_ask_human.sh
```

These runs produce question and patch-evaluation results together. Set each method's model variable to `claude-haiku-4-5` to run the second backbone.

## Token cost

Each method writes `cost.json` and `cost.jsonl` with input and output token counts. The paper reports mean usage per task; `cost.yaml` contains the configured token prices.
