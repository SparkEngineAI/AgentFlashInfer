import json
import os
import requests
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_parser():
    parser = argparse.ArgumentParser(description="request vllm serving.")
    parser.add_argument("--input-data-path", type=str, help="Specify the local input JSONL file path.")
    parser.add_argument("--output-path", type=str, default=None, nargs='?', help="Specify the turbo output path.")
    parser.add_argument("--max-sampling-len", type=int, default=65536, nargs='?', help="Specify the max output length.")
    parser.add_argument("--num-threads", type=int, default=8, nargs='?', help="num threads")
    parser.add_argument("--timeout-per-epoch", type=int, default=None, nargs='?', help="Timeout per eval epoch")
    parser.add_argument("--timeout-of-request", type=int, default=None, nargs='?', help="Timeout of a request")
    parser.add_argument("--eos-id", type=int, default=45892, nargs='?', help="stop token id")
    parser.add_argument("--think", type=int, default=1, nargs='?', help="slow think")
    return parser


def read_jsonl(path):
    if path.startswith('s3://'):
        raise ValueError("Only local JSONL files are supported; download S3 data before running this script.")
    if not os.path.exists(path):
        return None

    data = [json.loads(item) for item in
          open(path, "r", encoding="utf-8")]
    return data


def request_model_response(data, args):
    line_break_stop_ids = [45892, 2, 13, 92006, 92040, 92050, 92052, 92060, 92061, 92065, 92070, 92071, 92073, 92076,
                           92083, 92087, 92089, 92090, 92110, 92127, 92130, 92136, 92138, 92139, 92161, 92162, 92165,
                           92178, 92190, 92191, 92196, 92205, 92212, 92216, 92232, 92249, 92264, 92265, 92267, 92269,
                           92279, 92280, 92282, 92292, 92293, 92295, 92302, 92303, 92306, 92307, 92319, 92325, 92331,
                           92336, 92337, 92342, 92346, 92356, 92359, 92373, 92377, 92390, 92395, 92402, 92411, 92421,
                           92430, 92433, 92459, 92461, 92472, 92492, 92497, 92502, 92509, 92524, 92531, 92547, 92584,
                           92591, 92612, 92613, 92628, 92634, 92644, 92652, 92658, 92673, 92674, 92685, 92686, 92695,
                           92704, 92706, 92710, 92716, 92721, 92793, 92795, 92796, 92843, 92846, 92849, 92878, 92891,
                           92898, 92905, 92915, 92916, 92921, 92943, 92952, 92954, 92975, 92982, 92994, 93008, 93014,
                           93017, 93018, 93036, 93054, 93064, 93071, 93085, 93091, 93097, 93103, 93128, 93131, 93137,
                           93163, 93187, 93190, 93192, 93199, 93210, 93246, 93249, 93259, 93266, 93277, 93324, 93381,
                           93383, 93385, 93396, 93426, 93436, 93479, 93487, 93493, 93533, 93548, 93550, 93561, 93563,
                           93594, 93606, 93609, 93611, 93616, 93619, 93633, 93637, 93690, 93717, 93719, 93731, 93734,
                           93756, 93807, 93839, 93868, 93879, 93893, 93898, 93904, 93910, 93919, 93924, 93946, 93953,
                           93958, 93981, 93984, 93988, 94031, 94034, 94119, 94127, 94166, 94184, 94220, 94246, 94280,
                           94288, 94299, 94305, 94307, 94317, 94369, 94389, 94391, 94439, 94442, 94467, 94473, 94474,
                           94483, 94549, 94573, 94601, 94622, 94648, 94651, 94656, 94660, 94666, 94679, 94685, 94698,
                           94714, 94797, 94818, 94837, 94899, 94902, 94914, 94934, 94935, 94967, 95013, 95014, 95049,
                           95124, 95145, 95147, 95180, 95229, 95238, 95253, 95270, 95328, 95330, 95409, 95419, 95449,
                           95491, 95529, 95534, 95536, 95599, 95602, 95672, 95683, 95690, 95717, 95767, 95814, 95920,
                           95940, 95983, 95986, 96004, 96191, 96248, 96282, 96345, 96437, 96459, 96514, 96594, 96600,
                           96661, 96672, 96674, 96710, 96711, 96717, 96718, 96732, 96749, 96766, 96774, 96900, 96931,
                           96932, 97024, 97026, 97065, 97115, 97155, 97157, 97182, 97183, 97185, 97220, 97251, 97336,
                           97388, 97399, 97424, 97436, 97554, 97698, 97779, 97820, 97879, 97957, 97960, 98051, 98136,
                           98143, 98184, 98237, 98258, 98376, 98424, 98475, 98504, 98566, 98676, 98696, 98719, 98720,
                           98737, 98748, 98766, 98920, 98980, 98998, 99007, 99053, 99142, 99206, 99214, 99219, 99258,
                           99676, 99769, 99925, 99932, 99952, 99959, 99967, 99983]

    if args.eos_id == 13:
        stop_ids = line_break_stop_ids
    elif args.eos_id in [2, 45892]:
        stop_ids = [2, 45892]
    else:
        stop_ids = [args.eos_id]

    headers = {
        'Content-Type': 'application/json'
    }
    payload = {
        "top_p": 0.8,
        "top_k": -1,
        "temperature": 1.0,
        "stop_token_ids": stop_ids,
        "max_tokens": args.max_sampling_len,
    }
    url = "http://0.0.0.0:8000/v1/completions"
    prefix = "[unused9]系统：[unused10][unused9]用户："
    if args.think:
        suffix = "[unused10][unused9]助手："
    else:
        suffix = " /no_think[unused10][unused9]助手："
    payload['prompt'] = prefix + data['prompt'] + suffix

    for attempt in range(3):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=1000)
            if response.status_code == 200:
                try:
                    response_text = response.json()["choices"][0]["text"]
                    break
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    response_text = f"Error: Failed to parse response - {response.text}"
                    break
            else:
                response_text = f"Error: HTTP {response.status_code} - {response.text}"
                if 400 <= response.status_code < 500:
                    break

        except requests.exceptions.RequestException as e:
            response_text = f"Error: RequestException - {e}"

        if attempt < 4:
            wait_time = 2 ** attempt
            time.sleep(wait_time)
    data['response'] = response_text
    print(data)
    
    return data

def main(args):
    dataset = read_jsonl(args.input_data_path)
    pre_saved_ids = set(data['id'] for data in (read_jsonl(os.path.join(args.output_path, "infer_result.json")) or []) if 'id' in data)
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        # 提交所有任务，并用字典记录 {future: original_index}
        data_idx = 0
        future_to_idx = {}
        for data in dataset:
            if 'id' in data and data['id'] in pre_saved_ids:
                continue
            future_to_idx[executor.submit(request_model_response, data, args)] = data_idx
            data_idx += 1

        data_num = len(future_to_idx)
        results = [None] * data_num

        for future in as_completed(future_to_idx, timeout=args.timeout_per_epoch):
            data_idx = future_to_idx[future]
            try:
                data = future.result()
            except Exception as e:
                data = dataset[data_idx]
                data['error'] = str(e)

            results[data_idx] = data

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    with open(os.path.join(args.output_path, "infer_result.json"), "w", encoding="utf-8") as f:
        for i in range(len(results)):
            json_line = json.dumps(results[i], ensure_ascii=False)
            f.write(json_line + "\n")
    
        
if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)
