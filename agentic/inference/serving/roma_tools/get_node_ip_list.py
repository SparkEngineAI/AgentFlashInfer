import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description="node ip help")
parser.add_argument('--mode', default='all')

args = parser.parse_args()

# 获取文件路径
json_file_path = os.getenv('GLOBAL_RANK_TABLE_FILE_PATH')

# 检查环境变量是否设置
if not json_file_path:
    print("环境变量 GLOBAL_RANK_TABLE_FILE_PATH 未设置")
    sys.exit(1)

# 尝试打开和读取 JSON 文件
while True:
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取文件 {json_file_path} 时出错: {e}")
        sys.exit(1)
    if data.get('status', "") != "completed":
        time.sleep(30)
    else:
        break

# 查找 ip 对应的 rank
all_prefill_ip = ""
all_decode_ip = ""
for group in data.get('server_group_list', []):
    if group.get('group_id') in ["0", "1"]:
        for server in group['server_list']:
            all_decode_ip = all_decode_ip + " " + server.get('server_ip')
    if group.get('group_id') in ["2"]:
        for server in group['server_list']:
            all_prefill_ip = all_prefill_ip + " " + server.get('server_ip')

all_ip = None
if args.mode == "all":
    all_ip = all_prefill_ip + all_decode_ip
elif args.mode == "prefill":
    all_ip = all_prefill_ip
elif args.mode == "decode":
    all_ip = all_decode_ip
    
if all_ip.startswith(" "):
    all_ip = all_ip[1:]

# 输出找到的 IP 地址
if all_ip:
    print(all_ip)
else:
    print("没有找到 group_id 为 0 的 rank")
    sys.exit(1)
