import os
import time
import uuid
import base64
import asyncio
import logging
import numpy as np
from io import BytesIO
from PIL import Image
from aiohttp import web
from typing import Optional

import bl_execution.execution
try:
    from .comfyui_algo_api import use_switch_change_node
except ImportError:
    # Fallback for direct execution where bl_interface is not treated as a package.
    from comfyui_algo_api import use_switch_change_node


class BLOutput:
    @staticmethod
    def _base64_to_str(img_list):
        for i in range(len(img_list)):
            img_list[i] = str(img_list[i], "utf-8")
        return img_list

    @staticmethod
    def _array_to_base64(images, format='PNG'):
        results = list()
        for (_, image) in enumerate(images):
            # to pil
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if img.mode == "RGBA":
                img = img.convert("RGB")

            # to base64
            img_buffer = BytesIO()
            img.save(img_buffer, format=format)
            byte_data = img_buffer.getvalue()
            base64_str = base64.b64encode(byte_data)

            results.append(base64_str)
        return results

    @classmethod
    def get_output(cls, output_type: str, cache: list):
        """
            自动获取cache中多层嵌套的输出结果，包括image，str等
            output_type(str): image, video, text
            cache(list): 节点的所有输出
            response(dict): 输出
        """
        result = None

        if output_type == "image":
            # image 输出节点可能包含多个tensor
            result = cls._base64_to_str(cls._array_to_base64(cache[0]))
        elif output_type == "video":
            pass
        elif output_type == "text":
            pass
        else:
            logging.warning(f"{output_type} not support")
            # raise ValueError(f"fe_params:output_type {output_type} is not support, please use 'image' or 'video' or 'text'")

        return result


class BLRoute:
    def __init__(self, prompt_server, args):
        self.routes: web.RouteTableDef = web.RouteTableDef()
        self._app: Optional[web.Application] = None
        self.prompt_server = prompt_server

        cache_type = bl_execution.execution.CacheType.CLASSIC
        if args.cache_lru > 0:
            cache_type = bl_execution.execution.CacheType.LRU
        elif args.cache_ram > 0:
            cache_type = bl_execution.execution.CacheType.RAM_PRESSURE
        elif args.cache_none:
            cache_type = bl_execution.execution.CacheType.NONE
        self.exec = bl_execution.execution.PromptExecutor(
            server=self.prompt_server, 
            cache_type=cache_type, cache_args={ "lru" : args.cache_lru, "ram" : args.cache_ram })

        self.number = 0
        self.log_prefix = "[Flow Engine] "

        logging.info(self.log_prefix + "Init BLRoute success")

    def setup_routes(self):
        @self.routes.post("/pid")
        async def bl_api_pid(request):
            return web.json_response({"pid": os.getpid()})
        
        @self.routes.post('/base')
        async def bl_api_route(request):
            json_data =  await request.json()
            json_data = self.prompt_server.trigger_on_prompt(json_data)

            switch_config = json_data.get("switch", {})
            if switch_config:
                if "algo" in switch_config and "switch" not in switch_config:
                    switch_flags = [
                        bool(value)
                        for key, value in switch_config.items()
                        if key.startswith("switch_")
                    ]
                    switch_config["switch"] = any(switch_flags)

                json_data["prompt"] = use_switch_change_node(switch_config, json_data["prompt"])

            if "number" in json_data:
                number = float(json_data['number'])
            else:
                number = self.number
                if "front" in json_data:
                    if json_data['front']:
                        number = -number

                self.number += 1

            if "prompt" in json_data:
                prompt = json_data["prompt"]
                prompt_id = str(json_data.get("prompt_id", uuid.uuid4()))

                partial_execution_targets = None
                if "partial_execution_targets" in json_data:
                    partial_execution_targets = json_data["partial_execution_targets"]

                valid = await bl_execution.execution.validate_prompt(prompt_id, prompt, partial_execution_targets)
                extra_data = {}
                if "extra_data" in json_data:
                    extra_data = json_data["extra_data"]

                if "client_id" in json_data:
                    extra_data["client_id"] = json_data["client_id"]
                
                if valid[0]:
                    outputs_to_execute = valid[2]
                    sensitive = {}
                    for sensitive_val in bl_execution.execution.SENSITIVE_EXTRA_DATA_KEYS:
                        if sensitive_val in extra_data:
                            sensitive[sensitive_val] = extra_data.pop(sensitive_val)
                    extra_data["create_time"] = int(time.time() * 1000)  # timestamp in milliseconds
                    bl_params = json_data.get("bl_params", {})
                    if not bl_params:
                        bl_params = json_data.get("fe_params", {})
                    response = await asyncio.to_thread(self._forward, prompt, prompt_id, bl_params, extra_data, outputs_to_execute)
                    return web.json_response(response)
                else:
                    logging.warning(f"{self.log_prefix} invalid prompt: {valid[1]}")
                    return web.json_response({"error": valid[1], "node_errors": valid[3]})
            else:
                error = {
                    "type": "no_prompt",
                    "message": "No prompt provided",
                    "details": "No prompt provided",
                    "extra_info": {}
                }
                return web.json_response({"error": error, "node_errors": {}}) 

    def _forward(self, prompt: dict, prompt_id: str, bl_params: dict, extra_data: dict, outputs_to_execute: dict):
        """API forward execute nodes

        Args:
            prompt (dict): input prompt dict
            prompt_id (str): uuid for one task
            bl_params (dict): bl_params
            extra_data (dict): extra_data
            outputs_to_execute (dict): outputs_to_execute

        Returns:
            dict: response
        """
        response = {
            "prompt_id": prompt_id,
            "status": "error",
            "result": {},
            "detail": None
        }

        if len(bl_params) == 0:
            logging.warning(f"{self.log_prefix} input bl_params is empty, exit")
            response['detail'] = f"input bl_params is empty, exit"
            return response

        logging.info(f"{self.log_prefix} Processing forward: job_id: {prompt_id}")
        logging.info(f"{self.log_prefix} bl_params: {bl_params}\n")

        # main execute code
        try:
            start_time = time.perf_counter()
            self.exec.execute(prompt=prompt, prompt_id=prompt_id, extra_data=extra_data, execute_outputs=outputs_to_execute)
            end_time = time.perf_counter()
            logging.info(f"{self.log_prefix} [PERF]: {end_time - start_time}")
        except Exception as e:
            response['detail'] = str(e)
            return response
        
        output_nodes = bl_params.get("output_nodes", [])
        output_type = bl_params.get("output_type", "")
        if isinstance(output_nodes, str):
            output_nodes = [output_nodes, ]

        rank = os.getenv("RANK")
        logging.info(f"rank: {rank}")
        if rank is None or str(rank) == "0":
            # get output result according to fe_params
            for node in output_nodes:
                pre_node, pre_node_idx = self._get_preview_node_index(prompt[node])
                cache_result = self.exec.caches.outputs.get(pre_node).outputs[pre_node_idx]

                response['result'][node] = BLOutput.get_output(output_type=output_type, cache=cache_result)

        logging.info(f"{self.log_prefix} Processing forward: job_id: {response.get('prompt_id')} completed.")

        response["status"] = "success"
        return response

    def _get_preview_node_index(self, node):
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2:
                return value
        return None

    def get_app(self):
        if self._app is None:
            self._app = web.Application()
            self.setup_routes()
            self._app.add_routes(self.routes)
        return self._app