# SmartSense — Industrial Anomaly Detection & Diagnosis Agent

> An agentic AI system that monitors industrial sensor streams, detects anomalies using machine learning, and autonomously diagnoses root causes using Gemini's function-calling API.

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://your-railway-url.up.railway.app)
[![Python](https://img.shields.io/badge/python-3.11+-blue)](https://www.python.org)
[![Django](https://img.shields.io/badge/django-5.0-green)](https://www.djangoproject.com)
[![Gemini](https://img.shields.io/badge/google-gemini-orange)](https://ai.google.dev)

---

## The Problem

Industrial equipment produces thousands of sensor readings per hour — temperature, vibration, power draw, and dozens more. Engineers can't watch dashboards 24/7, and silent failures cause costly unplanned downtime. Anomaly detection is a solved problem at the ML level, but the harder problem is **what comes next**: when a sensor spikes at 3 AM, *why* did it spike, *how serious* is it, and *what should the operator do*?

## The Solution

SmartSense is a two-layer AI system:

1. **ML Layer (Detection):** An Isolation Forest model trained on normal operating data flags anomalies in real time as sensor data streams in.
2. **Agentic AI Layer (Diagnosis):** When an anomaly is detected, a Gemini-powered agent autonomously investigates — it queries sensor history, checks for recurring patterns, calculates severity, and produces a structured diagnosis with root cause, urgency, and recommended action.

The result: instead of a binary alert ("anomaly!"), engineers get a one-paragraph diagnostic report with reasoning.

## Architecture

```
+---------------+      +------------------+     +------------------+     +---------------+
| Sensor Stream | ---> | Isolation Forest | --> | Gemini Agent +   | --> | Django UI +   |
| (simulated)   |      | (Scikit-learn)   |     | Function Calling |     | Live Dashboard|
+---------------+      +------------------+     +------------------+     +---------------+
                                                          |
                              +---------------------------+-----------------------------+
                              v                           v                             v
                    query_sensor_history()      get_recent_anomalies()         calculate_severity()
```

The agent doesn't just call an LLM — it has access to **tools** (Python functions) it can call to investigate. This is the difference between a chatbot and an agent: the agent decides what information it needs and fetches it autonomously.

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Backend | Django 5 | Familiar stack, fast development, batteries included |
| ML Model | Scikit-learn (Isolation Forest) | Unsupervised — no labelled anomaly data needed |
| AI Agent | Google Gemini (function calling) | Free tier with generous limits, native tool use, fast inference |
| Database | SQLite (PostgreSQL ready) | Zero-config dev, easy migration |
| Frontend | Vanilla JS + Chart.js | No build step, ships fast |
| Deployment | Railway | One-click Django deploys |
| Static files | WhiteNoise | Serves static files without nginx |

## Demo

Visit the live demo and watch the dashboard for ~30 seconds — sensors stream every 5 seconds, and roughly 1 in 10 readings is an anomaly. When one fires, the Gemini agent investigates in real time and the diagnosis appears in the anomaly log.

Click any anomaly to see the full agent report — root cause, severity, recommended action, agent reasoning, and the exact sensor readings at detection time.

## Local Setup

```bash
# 1. Clone and install
git clone https://github.com/your-username/smartsense.git
cd smartsense
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (get one free at aistudio.google.com/apikey)

# 3. Generate training data and train the model
python monitor/ml/generate_data.py
python monitor/ml/train_model.py

# 4. Set up the database
python manage.py migrate

# 5. Run the server
python manage.py runserver
# Visit http://127.0.0.1:8000
```

## Engineering Decisions

### Why Isolation Forest?

Isolation Forest works well for unsupervised anomaly detection because it isolates anomalies by random partitioning — anomalous points are isolated faster (shorter path in the tree) because they're rare and different from the bulk of the data.

Alternatives considered:
- **DBSCAN:** Density-based; struggles when normal data has varied density
- **Autoencoder:** Higher recall but needs deep learning infrastructure overkill for 3 features
- **One-class SVM:** Slower to train, similar performance on this scale

For 3 features and ~1000 data points, Isolation Forest is faster to train (sub-second), faster at inference (<5ms), and easier to retrain when new normal patterns emerge.

### Why an agent and not just an LLM call?

A single LLM call would force the model to guess context without data. The agent pattern lets Gemini **request specific information** as needed — "is this temperature spike part of a 6-hour trend?", "have we seen this anomaly type recently?" — and then reason over the actual data.

This produces more accurate, more specific diagnoses. It also keeps the prompt short (no need to stuff all possible context into one call) and naturally scales when we add new tools later (e.g., `lookup_maintenance_log`, `check_replacement_part_inventory`).

### Why Gemini for the diagnosis layer?

I needed an LLM with reliable function-calling support and structured output. Gemini 2.5 Flash offers:
- Native tool-use API (no LangChain wrapper needed)
- Generous free tier — important for student projects and demo deployments
- Fast inference (~2-3 seconds per diagnosis)
- Multi-turn tool calling that fits the investigation pattern naturally

The agent code is structured so swapping providers (Claude, OpenAI, local Llama via Ollama) is mostly changing the SDK calls — the prompt, tools, and loop logic are provider-agnostic.

### Why synchronous agent calls?

For the demo, agent diagnosis runs synchronously inside the Django request — simplest to implement, easy to debug. In production this would be a Celery task with a Redis broker so sensor ingestion isn't blocked by the agent's 2-5 second diagnosis loop.

## Model Performance

Trained on 1000 samples (950 normal, 50 anomalies):

```
              precision    recall  f1-score
Normal             1.00      1.00     1.00
Anomaly            1.00      1.00     1.00
```

Note: The simulated data has clear-cut anomalies (e.g., temperature jumping from 23°C to 50°C), which is why scores are perfect. With real-world noisy sensor data, expect ~80-90% recall and ~70-80% precision — and tune the `contamination` parameter based on your actual anomaly rate.

Recall on anomalies matters more than precision — a false negative (missing a real failure) is more costly than a false positive (false alarm an operator dismisses).

## What I'd Build Next

- **Real IoT integration:** Replace `simulate_reading` with an MQTT subscriber (Paho-MQTT) consuming live sensor streams
- **Per-sensor models:** Train separate Isolation Forests per equipment type for higher precision
- **Time-series context:** Add an LSTM-based detector alongside Isolation Forest for trend anomalies
- **Maintenance integration:** Add a `create_maintenance_ticket` tool so the agent can take action, not just diagnose
- **Async agent:** Move agent calls to Celery + Redis for high-throughput sensor ingestion
- **Human feedback loop:** Let operators mark diagnoses as helpful/unhelpful, use signal to fine-tune prompts

## What I Learned

- **Tool-use prompting is a skill.** The agent's quality depended heavily on the system prompt enforcing the investigation order (severity -> history -> recurrence -> diagnose). Without explicit step-by-step instructions, the agent skipped tools and gave vague answers.
- **Fallbacks are non-negotiable.** LLM APIs can rate-limit, fail, or return malformed JSON. The `_fallback_diagnosis` function ensures the dashboard never breaks even when the agent fails.
- **JSON-mode prompting via constraint, not parameter.** Getting reliable JSON output requires explicit instructions and post-processing (stripping markdown fences, fallback parsing).
- **The agent loop is just a while-loop.** Demystifying agent frameworks: the entire "agent" is ~50 lines of code that alternates between asking the LLM what to do, executing tools, and feeding results back. No LangChain needed.

---

**Built by Smriti Mishra** · [LinkedIn](https://linkedin.com/in/your-handle) · [GitHub](https://github.com/your-handle)
