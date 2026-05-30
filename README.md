# SentinelLLM — AI Security Testing Framework

An open-source platform for red-teaming and benchmarking large language models. SentinelLLM evaluates prompt injection, jailbreaks, and adversarial behavior across multiple LLMs with automated scoring, reporting, and defense recommendations.

---

## Why SentinelLLM

Most LLM security repos are:
- Toy demos (few payloads, no scoring)
- Academic evaluations (static datasets, limited models)
- Single-file scripts (no architecture or extensibility)

SentinelLLM is a modular platform with:
- Automated multi-model benchmarking
- LLM-as-judge scoring
- MITRE ATLAS + OWASP mapping
- Defense recommendations
- PDF + JSON reporting
- Streamlit dashboard
- Clean, extensible architecture

---

## Key Findings (Empirical Results)

| Model        | Injection | Jailbreak | Overall |
|--------------|-----------|-----------|---------|
| llama3.2:1b  | 37.5%     | 0.0%      | 18.8%   |
| llama3.1:8b  | 0.0%      | 0.0%      | 0.0%    |
| qwen2.5:3b   | 50.0%     | 25.0%     | 37.5%   |

Observations:
- Model size does not guarantee safety.
- Small models resist social-engineering jailbreaks but fail token-boundary attacks.
- Token smuggling is the most transferable attack category.
- Qwen2.5:3b is more permissive than Llama models at similar scales.

---

## Features

- 30 prompt injection payloads (6 categories)
- 30 jailbreak payloads (7 categories)
- LLM-as-judge scoring (Groq llama-3.3-70b-versatile)
- Multi-model benchmarking
- MITRE ATLAS mapping (AML.T0051, AML.T0054)
- OWASP LLM Top 10 alignment
- Defense recommendations with code snippets
- PDF + JSON report generation
- Streamlit dashboard
- Typer CLI (run, benchmark, health)

---

## Architecture

sentinellm/
├── core/
│   ├── llm_client.py
│   ├── attack_runner.py
│   ├── scorer.py
│   └── groq_judge.py
├── attacks/
│   ├── prompt_injection/
│   └── jailbreaks/
├── analysis/
│   ├── mitre_mapper.py
│   ├── defense_advisor.py
│   └── pdf_reporter.py
├── dashboard/
│   └── app.py
└── cli/
    └── sentinel.py

---

## Setup

Requirements: Python 3.12+, Ollama, Groq API key

```
git clone [https://github.com/yourusername/sentinellm](https://github.com/yourusername/sentinellm)
cd sentinellm
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull llama3.2:1b
ollama pull llama3.1:8b
ollama pull qwen2.5:3b

cp .env.example .env
# Add GROQ_API_KEY

python -m cli.sentinel health
```

---

## Usage

```
# Run attack suites
python -m cli.sentinel run --attack injection
python -m cli.sentinel run --attack jailbreak
python -m cli.sentinel run --attack all

# Benchmark models
python -m cli.sentinel benchmark

# Dashboard
streamlit run dashboard/app.py
```

---

## Roadmap

- Cross-model transferability matrix
- Adaptive attack generator (AutoFuzzer)
- RAG poisoning module
- FastAPI backend
- Docker deployment
- MkDocs documentation site

---

## References

MITRE ATLAS  
OWASP LLM Top 10  
JailbreakBench  
HarmBench  

Built by Sadhana — AI/ML student focusing on model security and adversarial robustness.
