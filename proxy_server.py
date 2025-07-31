import asyncio
import json
import logging
from typing import Dict, Any, Optional, List

import aiohttp
from aiohttp import web
import click
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
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
        try:
            candidate_url = f"{self.candidate_host}{path_qs}"
            covered_tokens = []
            all_comparisons = []
            
            while len(covered_tokens) < len(main_tokens):
                # Prepare request with partial assistant response if needed
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
                        matching_count = 0
                        start_pos = len(covered_tokens)
                        
                        for i in range(len(candidate_tokens)):
                            if start_pos + i < len(main_tokens) and candidate_tokens[i] == main_tokens[start_pos + i]:
                                matching_count += 1
                            else:
                                break
                        
                        # Store comparison data
                        if matching_count > 0:
                            all_comparisons.append({
                                'start_pos': start_pos,
                                'matching_count': matching_count,
                                'main_json': main_json,
                                'candidate_json': candidate_json
                            })
                        
                        # Update covered tokens - include at least one token even if no match
                        # to ensure progress and avoid infinite loops
                        if matching_count > 0:
                            covered_tokens.extend(main_tokens[start_pos:start_pos + matching_count])
                        else:
                            # No match, add the main model's token to continue
                            if start_pos < len(main_tokens):
                                covered_tokens.append(main_tokens[start_pos])
                        
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.error(f"Error parsing candidate response in iteration: {e}")
                        break
            
            # Log all comparisons
            if all_comparisons:
                self._log_all_comparisons(all_comparisons, main_tokens)
                
        except Exception as e:
            logger.error(f"Error in iterative candidate comparison: {e}")
    
    def _log_all_comparisons(self, all_comparisons: List[Dict], main_tokens: List[str]) -> None:
        """Log aggregated comparison results from multiple candidate requests"""
        try:
            total_matching = sum(comp['matching_count'] for comp in all_comparisons)
            logger.info("\n" + "="*80)
            logger.info(f"KL DIVERGENCE COMPARISON: {total_matching}/{len(main_tokens)} tokens matched across {len(all_comparisons)} requests")
            logger.info("="*80)
            
            for idx, comparison in enumerate(all_comparisons):
                start_pos = comparison['start_pos']
                matching_count = comparison['matching_count']
                main_json = comparison['main_json']
                candidate_json = comparison['candidate_json']
                
                if matching_count > 0:
                    logger.info(f"\nRequest {idx + 1} - Tokens {start_pos + 1}-{start_pos + matching_count}:")
                    
                    # Extract logprobs for this segment
                    main_logprobs = []
                    for choice in main_json.get('choices', []):
                        logprobs = choice.get('logprobs', {})
                        content_logprobs = logprobs.get('content', [])
                        main_logprobs.extend(content_logprobs)
                    
                    candidate_logprobs = []
                    for choice in candidate_json.get('choices', []):
                        logprobs = choice.get('logprobs', {})
                        content_logprobs = logprobs.get('content', [])
                        candidate_logprobs.extend(content_logprobs)
                    
                    # Log main model tokens
                    logger.info("Main model:")
                    main_tokens_str = []
                    for i in range(matching_count):
                        if start_pos + i < len(main_logprobs):
                            token_data = main_logprobs[start_pos + i]
                            selected_token = token_data.get('token', '')
                            main_tokens_str.append(repr(selected_token))
                            
                            # Format top logprobs with 3 decimal points
                            candidates = []
                            for candidate in token_data.get('top_logprobs', [])[:self.n_probs]:
                                candidates.append((
                                    candidate.get('id', -1),
                                    repr(candidate.get('token', '')),
                                    f"{candidate.get('logprob', 0.0):.3f}"
                                ))
                            
                            # logger.info(f"  {repr(selected_token)} : {candidates}")
                    logger.info(f"  Tokens: {' '.join(main_tokens_str)}")
                    
                    # Log candidate model tokens
                    logger.info("\nCandidate model:")
                    candidate_tokens_str = []
                    for i in range(matching_count):
                        if i < len(candidate_logprobs):
                            token_data = candidate_logprobs[i]
                            selected_token = token_data.get('token', '')
                            candidate_tokens_str.append(repr(selected_token))
                            
                            # Format top logprobs with 3 decimal points
                            candidates = []
                            for candidate in token_data.get('top_logprobs', [])[:self.n_probs]:
                                candidates.append((
                                    candidate.get('id', -1),
                                    repr(candidate.get('token', '')),
                                    f"{candidate.get('logprob', 0.0):.3f}"
                                ))
                            
                            # logger.info(f"  {repr(selected_token)} : {candidates}")
                    logger.info(f"  Tokens: {' '.join(candidate_tokens_str)}")
            
            logger.info("\n" + "="*80 + "\n")
            
        except Exception as e:
            logger.error(f"Error logging comparisons: {e}")
    
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
