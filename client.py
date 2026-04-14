import os
import io
import json
import time
import uuid
import base64
import logging
import binascii
import cv2 as cv
import numpy as np
from PIL import Image
import urllib.request
import urllib.parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ Client ] %(levelname)s %(lineno)d : %(message)s"
)
logger = logging.getLogger(__name__)


def del_file_prefix(folder_path, prefix):
    for file in os.listdir(folder_path):
        if file.startswith(prefix):
            tmp = os.path.join(folder_path, file)
            try:
                os.remove(tmp)
            except:
                logger.warning(f"Del file {tmp} failed")


class Client:
    def __init__(self, parser):
        self.parser = parser
        self.subparser = self.parser.add_subparsers()
        self.logger = logger

    def load_json(self, json_path: str, job_id: str = None):
        with open(json_path, "r", encoding="utf-8") as f:
            params = json.loads(f.read())

        if isinstance(params, list) and len(params) == 2:
            params, bl_params = params
            switch = {}
        elif isinstance(params, list) and len(params) == 3:
            params, switch, bl_params = params
        elif isinstance(params, dict):
            params["job_id"] = params.get("job_id", job_id)
            bl_params = params.get("bl_params", {})
            switch = params.get("switch", {})
            params = params.get("prompt", {})
        else:
            raise NotImplementedError(f"json format has not been implemented") 

        message = {"prompt": params, "job_id": job_id, "bl_params": bl_params, "switch": switch}

        return message

    def get_parser(self):
        return self.parser

    def add_algo(self, algo_name: str, exec_func, **kwargs):
        algo_parser = self.subparser.add_parser(algo_name)

        for key, value in kwargs.items():
            algo_parser.add_argument(f"--{key}", **value)
              
        # set json changer
        self.json_changer = exec_func
        algo_parser.set_defaults(func=self.send_req, json_changer=exec_func, save_prefix=algo_name)
    
    def send_req(self, args):
        args = vars(args)

        job_id = str(uuid.uuid4())
        print("job_id: ", job_id)

        message = self.load_json(args.get("params"), job_id)

        # insert input to json
        json_changer = args.get("json_changer")
        if json_changer is not None:
            json_changer(message, args)
        
        output_type = message["bl_params"].get("output_type", "image")

        message = json.dumps(message).encode('utf-8')

        req =  urllib.request.Request("{}/custom/base".format(args.get("url")), data=message)
        response = json.loads(urllib.request.urlopen(req).read())
        
        results = self.save_result(response, output_type, args.get("save_path"), args.get("save_prefix"), "png", with_alpha=False)
        print(f"results: {results}")
    
    def save_result(self, respond, output_type, save_path, save_prefix, file_type, with_alpha=False):
        if not respond.get("result"):
            raise ValueError("no result")

        results = []
        ind = 0
        for out_node, out_res in respond["result"].items():
            if output_type == "image":
                
                for res in out_res:
                    result = self.decode_base64_to_image(res, with_alpha=with_alpha)
                    while os.path.exists(os.path.join(save_path, f"{save_prefix}_{ind}.{file_type}")):
                        ind += 1
                        if ind > 10:
                            for file in os.listdir(save_path):
                                tmp = os.path.join(save_path, file)
                                if os.path.isfile(tmp):
                                    os.remove(tmp)
                    bts_save_path = os.path.join(save_path, f"{save_prefix}_{ind}.{file_type}")
                    result.save(bts_save_path)
                    results.append(bts_save_path)
            elif output_type == "text":
                results.append(out_res)
            elif output_type == "video":
                results.append(out_res)
        return results

    def decode_base64_to_image(self, encoding, with_alpha=False):
        if encoding.startswith("data:image"):
            content = encoding.split(";")[1]
            image_encoded = content.split(",")[1]
        else:
            image_encoded = encoding
        if not with_alpha:
            try:
                image_stream = io.BytesIO(base64.b64decode(image_encoded))
            except binascii.Error:
                return None
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            img = cv.imdecode(file_bytes, cv.IMREAD_UNCHANGED)
            if img is None:
                try:
                    image_stream = io.BytesIO(base64.b64decode(image_encoded))
                    rgb = Image.open(image_stream)
                except Exception:
                    return None
                return rgb
            else:
                height, width, channels = img.shape
                if channels == 4:
                    frame = cv.cvtColor(img, cv.COLOR_RGBA2BGRA)
                elif channels == 3:
                    frame = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                else:
                    raise ValueError("image channels not support")
                return Image.fromarray(frame)
        else:
            try:
                image = Image.open(io.BytesIO(base64.b64decode(image_encoded)))
                return image
            except binascii.Error:
                return None

def test_exec(message: dict, args: dict):
    import random
    message["prompt"]["6"]["inputs"]["text"] = args.get("txt")
    message["prompt"]["3"]["inputs"]["seed"] = random.randint(0, 1000000)

def gx_multiple_references(message: dict, args: dict):
    message['prompt']['75']['inputs']['text'] = args.get("prompt")
    message['prompt']['218']['inputs']['int_value'] = args.get("width")
    message['prompt']['219']['inputs']['int_value'] = args.get("height")
    message['prompt']['244']['inputs']['image'] = args.get("img_1")
    message['prompt']['32']['inputs']['image'] = args.get("img_2")
    message['prompt']['39']['inputs']['image'] = args.get("img_3")
    if bool(args.get("switch", False)):
        message['switch']['switch'] = True
        if bool(args.get("switch_1", False)):
            message['switch']['switch_1'] = True
        if bool(args.get("switch_2", False)):
            message['switch']['switch_2'] = True


def kb_flux_v2(message: dict, args: dict):
    assert not args['Tagger'] or args['Tagger_img'], "Tagger mode input error"
    assert not args['PULID'] or args['PULID_img'], "PULID mode input error"
    assert not args['ControlNet'] or args['ControlNet_img'], "ControlNet mode input error"
    assert not args['Redux'] or args['Redux_img'], "Redux mode input error"

    if args['Tagger']:
        message['prompt']['94']["inputs"]['image'] = args['Tagger_img']
    else:
        message['switch']['4']['enable'] = False
    if args['PULID']:
        message['prompt']['54']["inputs"]['image'] = args['PULID_img']
    else:
        message['switch']['3']['enable'] = False
    if args['ControlNet']:
        message['prompt']['97']["inputs"]['image'] = args['ControlNet_img']
    else:
        message['switch']['2']['enable'] = False
    if args['Redux']:
        message['prompt']['133']["inputs"]['image'] = args['Redux_img']
    else:
        message['switch']['6']['enable'] = False
    
    if args['Lora']:
        message['switch']['1']["enable"] = True
    else:
        message['switch']['1']['enable'] = False
    if args['Sample_2nd']:
        message['switch']['5']["enable"] = True
    else:
        message['switch']['5']['enable'] = False
    if args['Resolution_Upscale']:
        message['switch']['7']["enable"] = True
    else:
        message['switch']['7']['enable'] = False

def kb_flux_v2_yolo(message: dict, args: dict):
    message['switch']['2']['enable'] = True if args['ControlNet'] else False
    message['switch']['3']['enable'] = True if args['PULID'] else False
    message['switch']['4']['enable'] = True if args['Tagger'] else False
    message['switch']['5']['enable'] = True if args['Sample_2nd'] else False
    message['switch']['6']['enable'] = True if args['Redux'] else False

    message['prompt']['94']['inputs']['image'] = args['img']


def gx_makeup_transfer_prep(message: dict, args: dict):
    message['prompt']['27']['inputs']['image'] = args.get("img1")
    message['prompt']['29']['inputs']['image'] = args.get("img2")
    message['prompt']['41']['inputs']['fit'] = args.get("makeup_fit")

def gx_halloween_themed_photo_prep(message: dict, args: dict):
    if bool(args.get("switch", False)):
        message['switch']['switch'] = True
        message['prompt']['62']['inputs']['image'] = args.get("img1")
        message['prompt']['64']['inputs']['image'] = args.get("img2")
    else:
       message['switch']['switch'] = False
       message['prompt']['62']['inputs']['image'] = args.get("img1")
        

def ai_hug_concat_v3(message: dict, args: dict):
    message['prompt']['579']['inputs']['image'] = args.get("img_1")
    message['prompt']['581']['inputs']['image'] = args.get("img_2")
    message['prompt']['536']['inputs']['int_value'] = args.get("width")
    message['prompt']['537']['inputs']['int_value'] = args.get("height")

def _parse_command_line():
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(epilog="$ python client.py --url=http://0.0.0.0:6780 comfyui -p sample.json -i sample.png", formatter_class=RawDescriptionHelpFormatter)

    # public args
    parser.add_argument("--url", default="http://0.0.0.0:7893", type=str, help="The base server url")
    parser.add_argument("--params", default="/home/code/aigc/FlowEngine/default.json", type=str, help="the input parameter json file path")
    parser.add_argument("--save_path", default="./output", type=str, help="default save path")
    parser.add_argument("--loop_cnt", "-l", default=1, type=int, help="loop times")

    client = Client(parser=parser)

    client.add_algo(algo_name="test", exec_func=test_exec, txt={"type": str, "help": "input prompt"})

    client.add_algo(algo_name="kb_flux_v2",
                    exec_func=kb_flux_v2,
                    Tagger={"action": "store_true", "help": "whether open tagger or not"},
                    Tagger_img={"type": str, "help": "Tagger input image"},
                    Lora={"action": "store_true", "help": "whether open lora or not"},
                    PULID={"action": "store_true", "help": "whether open pulid or not"},
                    PULID_img={"type": str, "help": "PULID input image"},
                    ControlNet={"action": "store_true", "help": "whether open controlent or not"},
                    ControlNet_img={"type": str, "help": "ControlNet input image"},
                    Redux={"action": "store_true", "help": "whether open redux or not"},
                    Redux_img={"type": str, "help": "Redux input image"},
                    Sample_2nd={"action": "store_true", "help": "whether open sample or not"},
                    Resolution_Upscale={"action": "store_true", "help": "whether open upscale or not"}
                    )


    client.add_algo(algo_name="kb_flux_v2_yolo",
                    exec_func=kb_flux_v2_yolo,
                    img={"type": str, "help": "input image"},
                    Tagger={"action": "store_true", "help": "whether open tagger or not"},
                    PULID={"action": "store_true", "help": "whether open pulid or not"},
                    ControlNet={"action": "store_true", "help": "whether open controlent or not"},
                    Redux={"action": "store_true", "help": "whether open redux or not"},
                    Sample_2nd={"action": "store_true", "help": "whether open sample or not"}
                    )


    client.add_algo(algo_name="gx_makeup_transfer_prep",
                    exec_func=gx_makeup_transfer_prep,
                    img1={"type": str, "help": "input user image"},
                    img2={"type": str, "help": "input template image"},
                    makeup_fit={"type": str, "help": "crop_fill_letterbox"},
                    )

    client.add_algo(algo_name="gx_halloween_themed_photo_prep",
                    exec_func=gx_halloween_themed_photo_prep,
                    switch={"type": bool, "help": "switch"},
                    img1={"type": str, "help": "input user image"},
                    img2={"type": str, "help": "input template image"},
                    )

    client.add_algo(algo_name="ai_hug_concat_v3", 
                    exec_func=ai_hug_concat_v3, 
                    img_1={"type": str, "help": "input image1"},
                    img_2={"type": str, "help": "input img2"},
                    width={"type": int, "help": "width"}, 
                    height={"type": int, "help": "height"})

    return parser.parse_args()

if __name__ == "__main__":
    _args = _parse_command_line()

    if hasattr(_args, "func"):
        for i in range(_args.loop_cnt):
            start = time.time()
            logger.info(f"------->>> execute time: {i}")
            _args.func(_args)

            if i % 500 == 0 and i >= 500:
                del_file_prefix("./output", _args.save_prefix)
            logger.info(f"------->>> cost: {time.time() - start}")

