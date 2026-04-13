import os
import sys
import time
import logging
import asyncio
import threading

import folder_paths
from fe_nodes.comfy.cli_args import args
from fe_app.app.logger import setup_logger, print_startup_warnings
from fe_utils.pre_start import (
    apply_custom_paths, execute_prestartup_script, 
    run, setup_database, setup_env_variable,
    cleanup_temp
)

#NOTE: These do not do anything on core ComfyUI, they are for custom nodes.
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['DO_NOT_TRACK'] = '1'

setup_logger(log_level=args.verbose, use_stdout=args.log_stdout, parallel=args.parallel)

apply_custom_paths()
execute_prestartup_script()


if 'torch' in sys.modules:
    logging.warning("WARNING: Potential Error in code: Torch already imported, torch should never be imported before this point.")


import fe_execution.execution
from fe_execution.comfy_execution.progress import get_progress_state
from fe_execution.comfy_execution.utils import get_executing_context

import server

import fe_nodes.nodes
import fe_nodes.comfy.utils
import fe_nodes.comfy.model_management
from comfy_api import feature_flags

import fe_utils.comfyui_version
import fe_utils.hook_breaker_ac10a0
from fe_utils.protocol import BinaryEventTypes


def cuda_malloc_warning():
    device = fe_nodes.comfy.model_management.get_torch_device()
    device_name = fe_nodes.comfy.model_management.get_torch_device_name(device)
    cuda_malloc_warning = False
    if "cudaMallocAsync" in device_name:
        for b in fe_utils.cuda_malloc.blacklist:
            if b in device_name:
                cuda_malloc_warning = True
        if cuda_malloc_warning:
            logging.warning("\nWARNING: this card most likely does not support cuda-malloc, if you get \"CUDA error\" please run ComfyUI with: --disable-cuda-malloc\n")


def hijack_progress(server_instance):
    def hook(value, total, preview_image, prompt_id=None, node_id=None):
        executing_context = get_executing_context()
        if prompt_id is None and executing_context is not None:
            prompt_id = executing_context.prompt_id
        if node_id is None and executing_context is not None:
            node_id = executing_context.node_id
        fe_nodes.comfy.model_management.throw_exception_if_processing_interrupted()
        if prompt_id is None:
            prompt_id = server_instance.last_prompt_id
        if node_id is None:
            node_id = server_instance.last_node_id
        progress = {"value": value, "max": total, "prompt_id": prompt_id, "node": node_id}
        get_progress_state().update_progress(node_id, value, total, preview_image)

        server_instance.send_sync("progress", progress, server_instance.client_id)
        if preview_image is not None:
            # Only send old method if client doesn't support preview metadata
            if not feature_flags.supports_feature(
                server_instance.sockets_metadata,
                server_instance.client_id,
                "supports_preview_metadata",
            ):
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id,
                )

    fe_nodes.comfy.utils.set_progress_bar_global_hook(hook)



def start_flowengine(asyncio_loop=None):
    if args.temp_directory:
        temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
        logging.info(f"Setting temp directory to: {temp_dir}")
        folder_paths.set_temp_directory(temp_dir)
    cleanup_temp()

    if not asyncio_loop:
        asyncio_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(asyncio_loop)

    prompt_server = server.PromptServer(asyncio_loop, fe_mode=True)

    fe_utils.hook_breaker_ac10a0.save_functions()
    asyncio_loop.run_until_complete(fe_nodes.nodes.init_extra_nodes(
        init_custom_nodes=(not args.disable_all_custom_nodes) or len(args.whitelist_custom_nodes) > 0,
        init_api_nodes=not args.disable_api_nodes
    ))
    fe_utils.hook_breaker_ac10a0.restore_functions()

    cuda_malloc_warning()
    setup_database()

    prompt_server.add_routes()
    hijack_progress(prompt_server)

    if args.quick_test_for_ci:
        exit(0)

    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    call_on_start = None
    if args.auto_launch:
        def startup_server(scheme, address, port):
            import webbrowser
            if os.name == 'nt' and address == '0.0.0.0':
                address = '127.0.0.1'
            if ':' in address:
                address = "[{}]".format(address)
            webbrowser.open(f"{scheme}://{address}:{port}")
        call_on_start = startup_server

    async def start_all():
        await prompt_server.setup()
        await run(prompt_server, 
                  address=args.listen, 
                  port=args.port + int(os.getenv("RANK", 0)) if args.parallel else args.port, 
                  verbose=not args.dont_print_server, 
                  call_on_start=call_on_start)

    # Returning these so that other code can integrate with the ComfyUI loop and server
    return asyncio_loop, prompt_server, start_all




if __name__ == "__main__":
    setup_env_variable()
    import fe_utils.cuda_malloc

    # Running directly, just start ComfyUI.
    logging.info("Python version: {}".format(sys.version))
    logging.info("ComfyUI version: {}".format(fe_utils.comfyui_version.__version__))

    if sys.version_info.major == 3 and sys.version_info.minor < 10:
        logging.warning("WARNING: You are using a python version older than 3.10, please upgrade to a newer one. 3.12 and above is recommended.")


    event_loop, _, start_all_func = start_flowengine()
    try:
        x = start_all_func()
        print_startup_warnings()
        event_loop.run_until_complete(x)
    except KeyboardInterrupt:
        logging.info("\nStopped server")

    cleanup_temp()
