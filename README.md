# KL Divergence Proxy

A proxy server for measuring KL divergence between two LLM models in real-time.

## Overview

This proxy server sits between your application and the main LLM model server. It:
- Forwards all HTTP requests to the target model server
- For `/v1/chat/completions` endpoints, automatically adds an `n_probs` parameter to request logit values
- Sends queries to a candidate model for comparison (with limited max_tokens)
- Compares token outputs and logs logprobs only for matching tokens
- Helps identify where model outputs diverge

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to set:
- `target_host`: The URL of your main model server
- `candidate_host`: The URL of your candidate model server (default: http://localhost:8081)
- `n_probs`: Number of top logits to request (default: 10)
- `port`: Port for the proxy to listen on (default: 8088)
- `candidate_max_tokens`: Max tokens for candidate model queries (default: 8)

## Usage

Start the proxy server:

```bash
python proxy_server.py
```

Or with a custom config file:

```bash
python proxy_server.py --config my_config.yaml
```

## Example Request

Send requests to the proxy as you would to your model server:

```bash
curl http://localhost:8080/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"model": "modelname",
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

The proxy will:
1. Add `"n_probs": 10` (or your configured value) to the request
2. Forward to the main model and get the full response
3. Send a limited request (with `candidate_max_tokens`) to the candidate model
4. Compare tokens and log logprobs only for matching tokens

## Logprobs Output

When tokens match between the main and candidate models, the proxy logs:
- Token-by-token comparison showing matching tokens
- Top N logprobs for each matching token (formatted to 3 decimal places)
- Separate sections for main model and candidate model outputs

## Next Steps

- Calculate actual KL divergence metrics between matching tokens
- Support multiple candidate queries to check full responses
- Add monitoring and visualization dashboard
- Export metrics for analysis
