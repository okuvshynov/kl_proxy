import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

import aiohttp
from aiohttp import web
import click
import yaml
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def kl_divergence_lower_bound(logits1_top1k, logits2_top1k, indices1, indices2):
    # Find common indices
    common_indices = np.intersect1d(indices1, indices2)
    
    # Create mappings
    idx_map1 = {idx: i for i, idx in enumerate(indices1)}
    idx_map2 = {idx: i for i, idx in enumerate(indices2)}
    
    # Get logits for common classes
    common_logits1 = np.array([logits1_top1k[idx_map1[idx]] for idx in common_indices])
    common_logits2 = np.array([logits2_top1k[idx_map2[idx]] for idx in common_indices])
    
    # Compute softmax over just these logits
    probs1 = np.exp(common_logits1) / np.sum(np.exp(common_logits1))
    probs2 = np.exp(common_logits2) / np.sum(np.exp(common_logits2))
    
    # Compute KL over common support
    kl = np.sum(probs1 * np.log(probs1 / probs2))
    
    return kl

class ProxyServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_host = config['target_host']
        self.candidate_host = config.get('candidate_host', 'http://localhost:8081')
        self.n_probs = config.get('n_probs', 10)
        self.port = config.get('port', 8080)
        self.candidate_max_tokens = config.get('candidate_max_tokens', 8)
        self.kld_by_ctx_size = {}
        
    async def handle_chat_completions(self, request: web.Request) -> web.Response:
        """Handle /v1/chat/completions endpoint with n_probs injection"""
        try:
            # Read the request body
            body = await request.json()
            
            # Add n_probs parameter to the request
            body['n_probs'] = self.n_probs
            logger.info(f"Added n_probs={self.n_probs} to chat completion request")
            
            # Forward the modified request
            headers = {k: v for k, v in request.headers.items() 
                      if k.lower() not in ['host', 'content-length']}
            
            async with aiohttp.ClientSession() as session:
                # First, get response from main model
                main_url = f"{self.target_host}{request.path_qs}"
                
                async with session.post(
                    main_url,
                    json=body,
                    headers=headers
                ) as main_response:
                    main_body = await main_response.read()
                    main_status = main_response.status
                    main_headers = main_response.headers
                    
                    # Try to parse main response and send to candidate
                    try:
                        main_json = json.loads(main_body)
                        
                        # Extract main model tokens for comparison
                        main_tokens = []
                        for choice in main_json.get('choices', []):
                            logprobs = choice.get('logprobs', {})
                            content_logprobs = logprobs.get('content', [])
                            for token_data in content_logprobs:
                                main_tokens.append(token_data.get('token', ''))
                        
                        if main_tokens:
                            # Start iterative candidate requests
                            await self._iterative_candidate_comparison(
                                session, body, headers, request.path_qs, main_json, main_tokens
                            )
                                
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing main response: {e}")
                    
                    # Return the main model's response
                    return web.Response(
                        body=main_body,
                        status=main_status,
                        headers=main_headers
                    )
                    
        except Exception as e:
            logger.error(f"Error in chat completions handler: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    async def _iterative_candidate_comparison(
        self, session: aiohttp.ClientSession, original_body: Dict[str, Any], 
        headers: Dict[str, str], path_qs: str, main_json: Dict[str, Any], 
        main_tokens: List[str]
    ) -> None:
        """Iteratively query candidate model to cover entire main response"""

        prompt_tokens = main_json['usage']['prompt_tokens']

        try:
            candidate_url = f"{self.candidate_host}{path_qs}"
            covered_tokens = []
            all_comparisons = []
            
            while len(covered_tokens) < len(main_tokens):
                # Prepare request with partial assistant response if needed
                logger.info(f"covered vs total: {len(covered_tokens)} < {len(main_tokens)}")
                candidate_body = original_body.copy()
                candidate_body['max_tokens'] = self.candidate_max_tokens
                
                if covered_tokens:
                    # Add partial assistant response to messages
                    partial_content = ''.join(covered_tokens)
                    messages = candidate_body.get('messages', []).copy()
                    messages.append({
                        "role": "assistant",
                        "content": partial_content
                    })
                    candidate_body['messages'] = messages
                
                # Send request to candidate
                async with session.post(
                    candidate_url,
                    json=candidate_body,
                    headers=headers
                ) as candidate_response:
                    candidate_body_response = await candidate_response.read()
                    
                    try:
                        candidate_json = json.loads(candidate_body_response)
                        
                        # Extract candidate tokens
                        candidate_tokens = []
                        for choice in candidate_json.get('choices', []):
                            logprobs = choice.get('logprobs', {})
                            content_logprobs = logprobs.get('content', [])
                            for token_data in content_logprobs:
                                candidate_tokens.append(token_data.get('token', ''))
                        
                        if not candidate_tokens:
                            break
                        
                        # Find matching tokens from current position
                        start_pos = len(covered_tokens)
                        
                        for i in range(len(candidate_tokens)):
                            if start_pos + i < len(main_tokens):
                                main_logprobs = main_json['choices'][0]['logprobs']['content'][start_pos + i]['top_logprobs']
                                candidate_logprobs = candidate_json['choices'][0]['logprobs']['content'][i]['top_logprobs']

                                all_comparisons.append((prompt_tokens + start_pos + i, main_logprobs, candidate_logprobs))
                                
                                covered_tokens.append(main_tokens[start_pos + i])
                                if candidate_tokens[i] != main_tokens[start_pos + i]:
                                    break
                            else:
                                break
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing candidate response in iteration: {e}")
                        break
                        
            for pos, main, candidate in all_comparisons:
                indices_a = [lp['id'] for lp in main]
                logits_a = [lp['logprob'] for lp in main]
                indices_b = [lp['id'] for lp in candidate]
                logits_b = [lp['logprob'] for lp in candidate]

                kld = kl_divergence_lower_bound(logits_a, logits_b, indices_a, indices_b)
                if pos not in self.kld_by_ctx_size:
                    self.kld_by_ctx_size[pos] = []
                
                self.kld_by_ctx_size[pos].append(kld)
                #logger.info(f'kld @ {pos} -> {kld}')

            for ctx_pos in sorted(self.kld_by_ctx_size.keys()):
                logger.info(f'kld[{ctx_pos}] = {np.average(self.kld_by_ctx_size[ctx_pos])}')

                
        except Exception as e:
            logger.error(f"Error in iterative candidate comparison: {e}")

    
    async def handle_generic_request(self, request: web.Request) -> web.Response:
        """Handle all other requests by forwarding them unchanged"""
        try:
            # Forward the request as-is
            headers = {k: v for k, v in request.headers.items() 
                      if k.lower() not in ['host', 'content-length']}
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.target_host}{request.path_qs}"
                body = await request.read()
                
                method = request.method.lower()
                method_func = getattr(session, method)
                
                async with method_func(
                    url,
                    data=body if body else None,
                    headers=headers
                ) as response:
                    # Stream the response back
                    response_body = await response.read()
                    
                    return web.Response(
                        body=response_body,
                        status=response.status,
                        headers=response.headers
                    )
                    
        except Exception as e:
            logger.error(f"Error in generic request handler: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    def create_app(self) -> web.Application:
        """Create the aiohttp application"""
        app = web.Application()
        
        # Route for chat completions
        app.router.add_post('/v1/chat/completions', self.handle_chat_completions)
        
        # Catch-all route for other requests
        app.router.add_route('*', '/{path:.*}', self.handle_generic_request)
        
        return app
    
    async def start(self):
        """Start the proxy server"""
        app = self.create_app()
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"Proxy server started on port {self.port}")
        logger.info(f"Main model: {self.target_host}")
        logger.info(f"Candidate model: {self.candidate_host}")
        logger.info(f"n_probs: {self.n_probs}, candidate_max_tokens: {self.candidate_max_tokens}")
        
        # Keep the server running
        await asyncio.Event().wait()


@click.command()
@click.option('--config', '-c', default='config.yaml', help='Path to configuration file')
def main(config):
    """Start the KL divergence proxy server"""
    # Load configuration
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file {config} not found")
        return
    
    # Create and start the server
    server = ProxyServer(config_data)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == '__main__':
    main()
