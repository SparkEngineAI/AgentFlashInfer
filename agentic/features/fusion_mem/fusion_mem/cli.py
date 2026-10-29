import argparse
import sys
import os
from fusion_mem.server.main import start_server
# 引入刚才写的客户端模块
from fusion_mem.client import MemoryClient

def main():
    parser = argparse.ArgumentParser(description="Fusion Mem 命令行工具")
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    # 1. Server 命令
    # 用法: fusion_mem server
    server_parser = subparsers.add_parser('server', help='启动 Fusion Mem Server')

    # 2. Save 命令
    # 用法: fusion_mem save "这是要保存的内容"
    save_parser = subparsers.add_parser('save', help='向 Server 保存数据')
    save_parser.add_argument('content', type=str, help='需要保存的文本内容')

    # 3. Search 命令
    # 用法: fusion_mem search "关键词"
    search_parser = subparsers.add_parser('search', help='从 Server 搜索数据')
    search_parser.add_argument('query', type=str, help='搜索关键词')

    args = parser.parse_args()
    host = os.environ.get("FUSION_MEM_HOST", "127.0.0.1")
    port = int(os.environ.get("FUSION_MEM_PORT", 15002))
    client = MemoryClient(f"http://{host}:{port}")

    # 路由分发
    if args.command == 'server':
        start_server()
    elif args.command == 'save':
        client.add([{"content": args.content ,"role": "user"}])
    elif args.command == 'search':
        print(f"response: {client.search(args.query)}")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()