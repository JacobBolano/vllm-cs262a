# SPDX-License-Identifier: Apache-2.0

import os

import aiohttp

import itertools

from quart import Quart, make_response, request

from vllm.logger import init_logger

logger = init_logger(__name__)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

# Create round-robin iterators for prefill and decode instances
prefill_servers = itertools.cycle(['http://localhost:8100/v1/completions'])
decode_servers = itertools.cycle(['http://localhost:8101/v1/completions',
                                  'http://localhost:8102/v1/completions', 
                                  'http://localhost:8200/v1/completions'])


async def forward_request(url, data):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }
        async with session.post(url=url, json=data,
                                headers=headers) as response:
            if response.status == 200:
                # if response.headers.get('Transfer-Encoding') == 'chunked':
                if True:
                    async for chunk_bytes in response.content.iter_chunked(
                            1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route('/v1/completions', methods=['POST'])
async def handle_request():
    try:
            
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request['max_tokens'] = 1

        # Select the next prefill server in the rotation
        # prefill_server = next(prefill_servers)
        # logger.info(f"Using prefill server: {prefill_server}")

        # finish prefill
        async for _ in forward_request('http://localhost:8100/v1/completions', prefill_request):
            continue

        # Select the next decode server in the rotation
        decode_server = next(decode_servers)
        logger.info(f"Using decode server: {decode_server}")

        # return decode
        # generator = forward_request(decode_server, original_request_data)
        generator = forward_request(decode_server,
                                    original_request_data)
        response = await make_response(generator)
        response.timeout = None

        return response

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


if __name__ == '__main__':
    app.run(port=8000)
