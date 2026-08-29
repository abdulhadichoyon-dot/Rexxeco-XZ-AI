# Rexxeco Learning Engine 🧠

A from-scratch learning engine prototype designed to evolve into a real self-learning AI system.

## Current architecture

```text
rexxeco-learning-engine/
├── core/
│   ├── learner.js
│   ├── evaluator.js
│   ├── retriever.js
│   └── memory.js
├── database/
│   ├── knowledge/
│   ├── memories/
│   └── feedback/
├── ai/
│   ├── model.js
│   └── prompts.js
├── api/
│   └── server.js
├── config/
│   └── config.js
├── tests/
├── package.json
├── .gitignore
└── README.md
```

## What it does now

- Accepts new knowledge through `POST /learn`
- Performs basic validation and confidence scoring
- Stores accepted memories locally
- Retrieves memories using simple token overlap
- Exposes a small HTTP API
- Keeps the model layer independent from external AI APIs

## Run

```bash
npm test
npm start
```

Server: `http://localhost:3000`

### Learn

```bash
curl -X POST http://localhost:3000/learn \
  -H "Content-Type: application/json" \
  -d '{"content":"JavaScript is a programming language.","source":"lesson-1","tags":["programming"]}'
```

### Retrieve

```bash
curl "http://localhost:3000/retrieve?q=javascript"
```

## Roadmap

1. Better evaluator
2. Structured knowledge
3. Feedback and correction system
4. Neural-network learning core
5. Tokenizer and text representation
6. Training pipeline
7. Model persistence
8. Conflict detection
9. Controlled self-learning loop

> This is an educational prototype. It is not yet a general-purpose autonomous AI.
