import logging
import os
import sys
import random
import numpy as np
import shutil 
from PIL import Image, ImageOps
from io import BytesIO
import base64

def update_command_args(command_args):
    for i in range(len(command_args)):
        if command_args[i] == "--sfast":
            sys.argv.append("--sfast")
        if command_args[i] == "--effect":
            sys.argv.append("--effect")
        # if command_args[i] == "--ckpt" and i + 1 < len(command_args):
        #     sys.argv.append(command_args[i])
        #     sys.argv.append(command_args[i+1])
        #     i += 1

# aisticker
def update_params_lora(params, loras):
    if len(loras) < 1:
        return params

    start_index = 50000
    start_node, _ = get_prompt_item(params["prompt"], "CheckpointLoaderSimple")

    for i in range(len(loras)):
        lora = loras[i]
        lora["model"] = [start_node, 0]
        lora["clip"] = [start_node, 1]
        item = {"inputs": lora, "class_type": "LoraLoader"}

        params["prompt"][str(start_index)] = item

        start_node = str(start_index)
        start_index += 1
    
    next_node, _ = get_prompt_items(params["prompt"], "CLIPTextEncode")
    for key in next_node:
        params["prompt"][key]["inputs"]["clip"][0] = start_node

    next_node, _ = get_prompt_items(params["prompt"], "KSampler")
    for key in next_node:
        params["prompt"][key]["inputs"]["model"][0] = start_node

    return params


def update_prompt_cycle(params, prompt, negative):
    prompt_key, _ = get_prompt_item_with_title(params["prompt"], "picture_book_prompt")
    negative_key, _ = get_prompt_item_with_title(params["prompt"], "picture_book_negative")

    prompt_len = len(prompt.split(params["prompt"][prompt_key]["inputs"]["delimiter"]))
    negative_len = len(negative.split(params["prompt"][negative_key]["inputs"]["delimiter"]))
    if prompt_len != negative_len:
        raise RuntimeError(f"config error: pic-book prompt and negative prompt length not match.")

    logging.warning("picture book num: {}".format(prompt_len))
    params["prompt"][prompt_key]["inputs"]["text"] = prompt
    params["prompt"][negative_key]["inputs"]["text"] = negative

    key, _ = get_prompt_item(params["prompt"], "ForInnerStart")
    params["prompt"][key]["inputs"]["i"] = 0
    params["prompt"][key]["inputs"]["total"] = prompt_len - 1
    params["prompt"][key]["inputs"]["stop"] = 1

    return params

def update_charactor_prompt(params, cht_prompt, cht_negative):
    prompt_key, _ = get_prompt_item_with_title(params["prompt"], "cht_prompt")
    negative_key, _ = get_prompt_item_with_title(params["prompt"], "cht_negative")
    params["prompt"][prompt_key]["inputs"]["text"] = cht_prompt
    params["prompt"][negative_key]["inputs"]["text"] = cht_negative

    return params

def update_milehigh_styler(params, styler):
    keys, _ = get_prompt_items(params["prompt"], "MilehighStyler")

    for i in range(len(keys)):
        key = keys[i]

        params["prompt"][key]["inputs"]["milehigh"] = styler

    return params

def update_prompt_seed(params):
    params_seed = {}
    for key in params:
        if "seed" in params[key]["inputs"] and isinstance(params[key]["inputs"]["seed"], int):
            if params[key]["inputs"]["seed"] <= -1:
                params[key]["inputs"]["seed"] = random.randint(1, 0xffffffff)

            params_seed[key] = params[key]["inputs"]["seed"]
    return params_seed, params

def update_video_path(params, path):
    key, _ = get_prompt_item(params["prompt"], "VHS_LoadVideoPath")

    if os.path.exists(path):
        params["prompt"][key]["inputs"]["video"] = path
    else:
        raise RuntimeError(f"config error: video not exist.")

    return params

def base64_to_str(img_list):
    for i in range(len(img_list)):
        img_list[i] = str(img_list[i], "utf-8")
    return img_list

def pil_to_base64(images, format = 'PNG'):
    results = list()
    for (_, image) in enumerate(images):
        # to base64
        img_buffer = BytesIO()
        image.save(img_buffer, format=format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)

        results.append(base64_str)
    return results

def array_to_base64(images, format='PNG'):
    results = list()
    for (_, image) in enumerate(images):
        # to pil
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        # to base64
        img_buffer = BytesIO()
        img.save(img_buffer, format=format)
        byte_data = img_buffer.getvalue()
        base64_str = base64.b64encode(byte_data)

        results.append(base64_str)
    return results

def update_params_lora(params, loras):
    if len(loras) < 1:
        return params

    start_index = 50000
    start_node, _ = get_prompt_item(params["prompt"], "CheckpointLoaderSimple")

    for i in range(len(loras)):
        lora = loras[i]
        lora["model"] = [start_node, 0]
        lora["clip"] = [start_node, 1]
        item = {"inputs": lora, "class_type": "LoraLoader"}

        params["prompt"][str(start_index)] = item

        start_node = str(start_index)
        start_index += 1

    next_node, _ = get_prompt_items(params["prompt"], "CLIPTextEncode")
    for key in next_node:
        params["prompt"][key]["inputs"]["clip"][0] = start_node

    next_node, _ = get_prompt_items(params["prompt"], "KSampler")
    for key in next_node:
        params["prompt"][key]["inputs"]["model"][0] = start_node

    return params

def save_image(images):
    save_dir = "output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for (batch_num, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        if i.ndim == 4:
            i = i.squeeze(axis = 0)
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save("{}/{}.png".format(save_dir, batch_num))

def get_prompt_item(prompt, obj):
    for key, item in prompt.items():
        if item["class_type"] == obj:
            return key, item
    return -1, None

def get_prompt_item_with_title(prompt, obj):
    for key, item in prompt.items():
        if "_meta" not in item:
            continue
        if item["_meta"]["title"] == obj:
            return key, item
    return -1, None

def get_prompt_items(prompt, obj):
    keys = []
    items = []

    for key, item in prompt.items():
        if item["class_type"] == obj:
            keys.append(key)
            items.append(item)
    return keys, items

def get_prompt_items_result(prompt, obj):
    result = []
    for key, item in prompt.items():
        if item["class_type"] == obj:
            result.append((key, item))
    return result

def get_preview_images(prompt, result, title = None):
    items = get_prompt_items_result(prompt, "PreviewImage")
    output = []
    for item in items:
        if title != None:
            item_title = item[1]["_meta"]["title"]
            if item_title != title:
                continue
        key = item[1]["inputs"]["images"][0]
        index = item[1]["inputs"]["images"][1]
        output += result[key][index][0]
        
    return output

def save_image_to_dir(images, save_dir = "output"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for (batch_num, image) in enumerate(images):
        i = 255. * image.cpu().numpy()
        if i.ndim == 4:
            i = i.squeeze(axis = 0)
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save("{}/{}.png".format(save_dir, batch_num))

# mx liveportrait json处理函数
def mx_livep_merge(json_params,use_video=False):
    
    if use_video:
        json_params["24"]["inputs"] = {
            "yield_num": json_params["24"]["inputs"]["yield_num"],
            "YIELD":["31", 0]
        }
    else:
        json_params["24"]["inputs"] = {
            "yield_num": json_params["24"]["inputs"]["yield_num"],
            "image":["15", 0]
        }
    
    return json_params

# mx 换脸 json处理函数
def mx_face_swap_video_or_image(json_params,video=False):
    
    if video:
        del json_params['95']
        del json_params['92']
        del json_params['94']
        
    else:
        del json_params['16']
        del json_params['77']
        del json_params['5']
    
    return json_params

def pic_sdxl_common_switch(switch, json_params):
    json_params['14']['inputs']['model'][0] = "26"
    del json_params['7']
    del json_params['8']
    del json_params['9']
    del json_params['10']
    del json_params['16']
    del json_params['17']
    del json_params['18']
    del json_params['20']
    del json_params['24']
    json_params['25']['inputs']['images'][0] = "14"
    json_params['25']['inputs']['images'][1] = 5
    del json_params['29']
    return json_params

def gx_multiple_references(switch, json_params):
    if not switch['switch']:
        return json_params
    if switch['switch_1'] and switch['switch_2']:
        del json_params['24']
        del json_params['27']
        del json_params['32']
        del json_params['35']
        del json_params['37']
        del json_params['39']
        json_params['40']['inputs']['conditioning'][0] = "41"
        del json_params['51']
        del json_params['52']
        del json_params['63']
        del json_params['93']
        del json_params['94']['inputs']["text_3"]
        del json_params['94']['inputs']["text_4"]
        del json_params['162']
        del json_params['164']
        return json_params
    elif not switch['switch_1'] and switch['switch_2']:
        del json_params['24']
        del json_params['37']
        del json_params['39']
        json_params['40']['inputs']['conditioning'][0] = "35"
        del json_params['51']
        del json_params['63']
        del json_params['94']['inputs']["text_4"]
        del json_params['164']
        return json_params
    elif switch['switch_1'] and not switch['switch_2']:
        json_params['24']['inputs']['conditioning'][0] = "41"
        del json_params['27']
        del json_params['32']
        del json_params['35']
        del json_params['52']
        del json_params['93']
        del json_params['94']['inputs']["text_3"]
        del json_params['162']
        return json_params
    else:
        raise NotImplementedError(f"Algo:gx_multiple_references, This switch has not been implemented")

def qwen_ie2509_aiphoto(switch, json_params):
    if not switch['switch']:
        return json_params
    if switch['switch_1'] and switch['switch_2']:
        del json_params['32']['inputs']["image2"]
        del json_params['32']['inputs']["image3"]
        del json_params['110']
        del json_params['113']
        del json_params['114']
        del json_params['117']
        del json_params['122']
        del json_params['123']
        return json_params
    elif not switch['switch_1'] and switch['switch_2']:
        del json_params['32']['inputs']["image3"]
        del json_params['113']
        del json_params['122']
        return json_params
    elif switch['switch_1'] and not switch['switch_2']:
        del json_params['32']['inputs']["image2"]
        del json_params['110']
        del json_params['114']
        del json_params['117']
        del json_params['123']
        return json_params
    else:
        raise NotImplementedError(f"Algo:qwen_ie2509_aiphoto, This switch has not been implemented")

def qwen_ie2509_general_v1(switch, json_params):
    if not switch['switch']:
        return json_params
    if switch['switch_1'] and switch['switch_2'] and not switch['switch_3']:
        del json_params['3']
        del json_params['15']['inputs']["image3"]
        return json_params
    elif switch['switch_1'] and not switch['switch_2'] and not switch['switch_3']:
        del json_params['2']
        del json_params['3']
        del json_params['15']['inputs']["image2"]
        del json_params['15']['inputs']["image3"]
        return json_params
    elif not switch['switch_1'] and switch['switch_2'] and switch['switch_3']:
        del json_params['10']
        json_params['12']['inputs']['model'][0] = "11"
        return json_params
    elif not switch['switch_1'] and switch['switch_2'] and not switch['switch_3']:
        del json_params['3']
        del json_params['15']['inputs']["image3"]
        del json_params['10']
        json_params['12']['inputs']['model'][0] = "11"
        return json_params
    elif not switch['switch_1'] and not switch['switch_2'] and not switch['switch_3']:
        del json_params['2']
        del json_params['3']
        del json_params['15']['inputs']["image2"]
        del json_params['15']['inputs']["image3"]
        del json_params['10']
        json_params['12']['inputs']['model'][0] = "11"
        return json_params
    elif switch['switch_1'] and switch['switch_2'] and switch['switch_3']:
        return json_params
    else:
        raise NotImplementedError(f"Algo:qwen_ie2509_aiphoto, This switch has not been implemented")

switch_algo_dict = {
    'pic_sdxl_common': pic_sdxl_common_switch,
    'gx_multiple_references': gx_multiple_references,
    'qwen_ie2509_aiphoto':qwen_ie2509_aiphoto,
    'qwen_ie2509_general_v1':qwen_ie2509_general_v1
}

def modify_nodes_with_group(groups, params):
    def check_node(item):
        if type(item) == list and len(item) == 2 and type(item[0]) == str and type(item[1]) == int:
            return True

        return False

    def find_preview(k, p_k):
        new_v = None
        new_p_k = None
        group = None
        new_k = None
        p_k_n = None
        # 找节点k前置节点所属组被删除时，对应的节点连接信息
        # (node1->group->node2, group被删除时node1和node2的连接信息)
        for _, g_value in groups.items():
            if not g_value["enable"] and k in g_value["modify"] and p_k in g_value["modify"][k]["inputs"]:
                new_v = g_value["modify"][k]["inputs"][p_k]
                # if not check_node(new_v):
                p_k_n = new_v[0]
                group = g_value
                break
        if new_v is None:
            return None, None, None, None

        # find node from preview connection in 'delete node'
        # (node1->group(obj_node, other_nodes)->node2, group被删除时找到obj_node信息)
        if group is not None:
            for cur_node_key in group["nodes"]:
                for cur_key, cur_val in params[str(cur_node_key)]["inputs"].items():
                    if new_v == cur_val:
                        new_k = cur_node_key
                        new_p_k = cur_key
                        break
                if new_k is not None:
                    break

        return str(new_k), new_v, str(p_k_n), new_p_k

    # recursively search for the previous node connection
    def recursive_fix(k, p_k):
        new_k, new_v, p_k_n, new_p_k = find_preview(k, p_k)
        if new_v is None:
            return None

        if p_k_n in new_params:
            return new_v
        
        return recursive_fix(new_k, new_p_k)

    new_params = params.copy()

    # delete nodes by group info
    for _, group in groups.items():
        if group["enable"]:
            continue
        for key in group["nodes"]:
            new_params.pop(str(key))

    # traversal all nodes and then fix
    for key, value in new_params.items():
        for input_key, input_value in value["inputs"].items():
            if not check_node(input_value):
                continue
            if input_value[0] in new_params:
                continue

            new_params[key]["inputs"][input_key] = recursive_fix(key, input_key)

    return new_params

def use_switch_change_node(switch, params):
    if 'algo' in switch and switch['algo'] in switch_algo_dict:
        return switch_algo_dict[switch['algo']](switch, params)
    # 新的多排列统一从这里走，ui全排列全部开启后，导出switch api的json即可直接用
    elif len(switch) > 0:
        return modify_nodes_with_group(switch, params)
    else:
        raise NotImplementedError(f"{switch['algo']} has not been implemented")