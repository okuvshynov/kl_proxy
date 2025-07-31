# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KL Divergence Proxy is a transparent HTTP proxy server designed to measure KL divergence between two LLM models in real-time. It intercepts requests to LLM endpoints and automatically adds logit probability parameters for model comparison.

## Key Commands

### Running the Server
```bash
python proxy_server.py                    # Start with default config.yaml
python proxy_server.py --config custom.yaml  # Start with custom config
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

## Architecture

The proxy server consists of:

1. **ProxyServer class** (proxy_server.py:19-128): Core proxy functionality
   - Handles HTTP request forwarding between client and target LLM server
   - Intercepts `/v1/chat/completions` requests to inject `n_probs` parameter
   - Forwards all other requests unchanged
   - Uses aiohttp for async HTTP handling

2. **Request Flow**:
   - Client → Proxy Server → Target LLM Server
   - For chat completion requests: automatically adds `n_probs` parameter
   - All other endpoints pass through transparently

3. **Configuration** (config.yaml):
   - `target_host`: Destination LLM server URL
   - `n_probs`: Number of top logits to request
   - `port`: Local proxy listening port

## Development Notes

- Uses async/await pattern throughout for concurrent request handling
- Logging configured to INFO level for debugging
- Click CLI framework for command-line interface
- YAML configuration for flexibility

## Future Implementation Areas

The README indicates planned features not yet implemented:
- Async forwarding to quantized model server for comparison
- Logit extraction and KL divergence calculation
- Metrics monitoring and visualization