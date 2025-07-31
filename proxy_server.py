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
        self.n_probs = config.get('n_probs', 10)
        self.port = config.get('port', 8080)
        
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
                url = f"{self.target_host}{request.path_qs}"
                
                async with session.post(
                    url,
                    json=body,
                    headers=headers
                ) as response:
                    # Read and parse the response
                    response_body = await response.read()
                    
                    # Try to parse and log logprobs if present
                    try:
                        response_json = json.loads(response_body)
                        self._log_logprobs(response_json)
                    except (json.JSONDecodeError, KeyError):
                        # If parsing fails, just pass through
                        pass
                    
                    return web.Response(
                        body=response_body,
                        status=response.status,
                        headers=response.headers
                    )
                    
        except Exception as e:
            logger.error(f"Error in chat completions handler: {e}")
            return web.Response(
                text=json.dumps({"error": str(e)}),
                status=500,
                content_type='application/json'
            )
    
    def _log_logprobs(self, response_json: Dict[str, Any]) -> None:
        """Parse and log logprobs from the response"""
        try:
            for choice in response_json.get('choices', []):
                logprobs = choice.get('logprobs', {})
                content_logprobs = logprobs.get('content', [])
                
                if content_logprobs:
                    logger.info("\n" + "="*80)
                    logger.info("LOGPROBS OUTPUT:")
                    logger.info("="*80)
                    
                    for token_data in content_logprobs:
                        selected_token = token_data.get('token', '')
                        selected_id = token_data.get('id', -1)
                        selected_logprob = token_data.get('logprob', 0.0)
                        
                        # Format top logprobs as list of tuples
                        candidates = []
                        for candidate in token_data.get('top_logprobs', [])[:self.n_probs]:
                            candidates.append((
                                candidate.get('id', -1),
                                repr(candidate.get('token', '')),  # Use repr to show escape sequences
                                candidate.get('logprob', 0.0)
                            ))
                        
                        # Log in the requested format
                        logger.info(f"{repr(selected_token)} : {candidates}")
                    
                    logger.info("="*80 + "\n")
                    
        except Exception as e:
            logger.error(f"Error parsing logprobs: {e}")
    
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
        logger.info(f"Forwarding requests to {self.target_host}")
        logger.info(f"n_probs parameter set to {self.n_probs}")
        
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
