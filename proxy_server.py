import asyncio
import json
import logging
from typing import Dict, Any, Optional

import aiohttp
from aiohttp import web
import click
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProxyServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_host = config['target_host']
        self.candidate_host = config.get('candidate_host', 'http://localhost:8081')
        self.n_probs = config.get('n_probs', 10)
        self.port = config.get('port', 8080)
        self.candidate_max_tokens = config.get('candidate_max_tokens', 8)
        
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
                        
                        # Prepare candidate request with limited tokens
                        candidate_body = body.copy()
                        candidate_body['max_tokens'] = self.candidate_max_tokens
                        
                        # Send request to candidate model
                        candidate_url = f"{self.candidate_host}{request.path_qs}"
                        async with session.post(
                            candidate_url,
                            json=candidate_body,
                            headers=headers
                        ) as candidate_response:
                            candidate_body_response = await candidate_response.read()
                            
                            try:
                                candidate_json = json.loads(candidate_body_response)
                                self._compare_and_log_logprobs(main_json, candidate_json)
                            except (json.JSONDecodeError, KeyError) as e:
                                logger.error(f"Error parsing candidate response: {e}")
                                
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
    
    def _compare_and_log_logprobs(self, main_json: Dict[str, Any], candidate_json: Dict[str, Any]) -> None:
        """Compare tokens from main and candidate models, log matching tokens' logprobs"""
        try:
            # Extract tokens from both responses
            main_tokens = []
            main_logprobs = []
            for choice in main_json.get('choices', []):
                logprobs = choice.get('logprobs', {})
                content_logprobs = logprobs.get('content', [])
                for token_data in content_logprobs:
                    main_tokens.append(token_data.get('token', ''))
                    main_logprobs.append(token_data)
            
            candidate_tokens = []
            candidate_logprobs = []
            for choice in candidate_json.get('choices', []):
                logprobs = choice.get('logprobs', {})
                content_logprobs = logprobs.get('content', [])
                for token_data in content_logprobs:
                    candidate_tokens.append(token_data.get('token', ''))
                    candidate_logprobs.append(token_data)
            
            # Find the number of matching tokens
            matching_count = 0
            for i in range(min(len(main_tokens), len(candidate_tokens))):
                if main_tokens[i] == candidate_tokens[i]:
                    matching_count += 1
                else:
                    break
            
            # Log only if there are matching tokens
            if matching_count > 0:
                logger.info("\n" + "="*80)
                logger.info(f"KL DIVERGENCE COMPARISON: {matching_count} matching tokens")
                logger.info("="*80)
                
                # Log main model tokens
                logger.info("Main model:")
                for i in range(matching_count):
                    token_data = main_logprobs[i]
                    selected_token = token_data.get('token', '')
                    
                    # Format top logprobs with 3 decimal points
                    candidates = []
                    for candidate in token_data.get('top_logprobs', [])[:self.n_probs]:
                        candidates.append((
                            candidate.get('id', -1),
                            repr(candidate.get('token', '')),
                            f"{candidate.get('logprob', 0.0):.3f}"
                        ))
                    
                    logger.info(f"  {repr(selected_token)} : {candidates}")
                
                # Log candidate model tokens
                logger.info("\nCandidate model:")
                for i in range(matching_count):
                    token_data = candidate_logprobs[i]
                    selected_token = token_data.get('token', '')
                    
                    # Format top logprobs with 3 decimal points
                    candidates = []
                    for candidate in token_data.get('top_logprobs', [])[:self.n_probs]:
                        candidates.append((
                            candidate.get('id', -1),
                            repr(candidate.get('token', '')),
                            f"{candidate.get('logprob', 0.0):.3f}"
                        ))
                    
                    logger.info(f"  {repr(selected_token)} : {candidates}")
                
                logger.info("="*80 + "\n")
                
        except Exception as e:
            logger.error(f"Error comparing logprobs: {e}")
    
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
