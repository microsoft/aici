from typing import List, Union, Dict, Any, Tuple, cast

from vllm import LLMEngine
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.protocol import ErrorResponse

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi import Request

from . import add_cli_args, runner_from_cli
from ._vllm_protocol import RunRequest, SetTagsRequest
from ._vllm_runner import AiciRunnerCompletion

app = api_server.app
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


def vllm_server_main():
    parser = api_server.make_arg_parser()
    add_cli_args(parser)
    args = parser.parse_args()

    engine = api_server.create_engine(args)

    model_config, served_model_names = api_server.get_model_config(
        args, engine)
    print("MODEL", served_model_names)
    dtype = str(model_config.dtype).replace("torch.", "").replace("float", "f")
    pyaici_runner = runner_from_cli(args, dtype=dtype)
    pyaici_runner.fast_api()

    global pyaici_runner_completion
    pyaici_runner_completion = AiciRunnerCompletion(pyaici_runner, engine,
                                                    model_config,
                                                    served_model_names,
                                                    args.lora_modules)
    assert isinstance(engine.engine, LLMEngine)
    # print("eos_token_id (vllm):", engine.engine._get_eos_token_id(None))
    engine.engine.sampling_controller = \
        pyaici_runner_completion.sampling_controller
    api_server.start_engine(args, engine)


if __name__ == "__main__":
    vllm_server_main()
