from typing import List, Union, Dict, Any, Tuple, cast

from vllm import LLMEngine
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.protocol import ErrorResponse
from vllm.entrypoints.logger import RequestLogger

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import Request

from . import add_cli_args, runner_from_cli
from ._vllm_protocol import RunRequest, SetTagsRequest
from ._vllm_runner import AiciRunnerCompletion

app = api_server.router
pyaici_runner_completion: AiciRunnerCompletion


def _no_aici():
    return JSONResponse({"error": "AICI runtime is not enabled"},
                        status_code=501)


@app.post("/v1/controllers")
async def upload_aici_module(request: Request):
    if not pyaici_runner_completion:
        return _no_aici()
    contents = await request.body()
    return JSONResponse(
        await
        pyaici_runner_completion.aici_runner.upload_module_async(contents))


@app.post("/v1/run")
async def aici_run(request: RunRequest, raw_request: Request):
    if not pyaici_runner_completion:
        return _no_aici()
    r = await pyaici_runner_completion.prep_completion(request)
    if isinstance(r, ErrorResponse):
        return JSONResponse(r.model_dump(), status_code=r.code)
    generator = pyaici_runner_completion.create_completion(r, raw_request)
    return StreamingResponse(content=generator, media_type="text/event-stream")


@app.post("/v1/controllers/tags")
async def aici_set_tags(request: SetTagsRequest):
    if not pyaici_runner_completion:
        return _no_aici()
    # non-admin users can only set tags that start with their username
    auto_info = {"user": "vllm", "is_admin": True}
    r = await pyaici_runner_completion.aici_runner.set_tags(
        request.module_id, request.tags, auth_info=auto_info)
    return JSONResponse(r)


@app.get("/v1/controllers/tags")
async def aici_get_tags():
    if not pyaici_runner_completion:
        return _no_aici()
    r = await pyaici_runner_completion.aici_runner.get_tags()
    return JSONResponse(r)

from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser
import vllm.entrypoints.openai.api_server as api_server
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.usage.usage_lib import UsageContext
from typing import Optional
import asyncio


def get_model_config(engine: AsyncLLMEngine):
    event_loop: Optional[asyncio.AbstractEventLoop]
    try:
        event_loop = asyncio.get_running_loop()
    except RuntimeError:
        event_loop = None

    if event_loop is not None and event_loop.is_running():
        # If the current is instanced by Ray Serve,
        # there is already a running event loop
        model_config = event_loop.run_until_complete(engine.get_model_config())
    else:
        # When using single vLLM without engine_use_ray
        model_config = asyncio.run(engine.get_model_config())

    return model_config


def vllm_server_main():
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server with AG2 support."
    )
    parser = make_arg_parser(parser)
    add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.OPENAI_API_SERVER
    )

    if args.served_model_name is not None:
        served_model_names = args.served_model_name
    else:
        served_model_names = [args.model]
    print("MODEL", served_model_names)

    model_config = get_model_config(engine)
    dtype = str(model_config.dtype).replace("torch.", "").replace("float", "f")

    pyaici_runner = runner_from_cli(args, dtype=dtype)
    pyaici_runner.fast_api()

    # n_vocab = model_config.get_vocab_size()

    if args.disable_log_requests:
        request_logger = None
    else:
        request_logger = RequestLogger(max_log_len=args.max_log_len)

    global pyaici_runner_completion
    pyaici_runner_completion = AiciRunnerCompletion(
        pyaici_runner,
        engine,
        model_config,
        served_model_names,
        lora_modules=args.lora_modules,
        prompt_adapters=args.prompt_adapters,
        request_logger=request_logger,
    )
    assert isinstance(engine.engine, LLMEngine)
    # print("eos_token_id (vllm):", engine.engine._get_eos_token_id(None))
    engine.engine.sampling_controller = pyaici_runner_completion.sampling_controller

    api_server.run_server(args, llm_engine=engine)


if __name__ == "__main__":
    vllm_server_main()
