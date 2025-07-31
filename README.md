# KL Divergence Proxy

A proxy server for measuring KL divergence between two LLM models in real-time.

## Overview

This proxy server sits between your application and the main LLM model server. It:
- Forwards all HTTP requests to the target model server
- For `/v1/chat/completions` endpoints, automatically adds an `n_probs` parameter to request logit values
- (Future) Asynchronously sends queries to a quantized model for comparison
- (Future) Calculates KL divergence between the two models' outputs

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to set:
- `target_host`: The URL of your main model server
- `n_probs`: Number of top logits to request (default: 10)
- `port`: Port for the proxy to listen on (default: 8080)

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

The proxy will automatically add `"n_probs": 10` (or your configured value) to the request before forwarding it.

## Next Steps

- Add async forwarding to quantized model server
- Implement logit extraction and comparison
- Calculate and report KL divergence metrics
- Add monitoring and visualization
