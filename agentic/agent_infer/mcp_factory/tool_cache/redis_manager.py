import os
import sys
import time
import signal
import subprocess
import atexit
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import config.config as config

_TOOL_CACHE_SWITCH = config.tool_cache_switch  # 工具缓存总开关

class RedisManager:
    def __init__(self, redis_bin_path=config.REDIS_BIN_PATH, port=config.REDIS_PORT, max_memory=config.REDIS_MAX_MEMORY, conf_dir=config.REDIS_CONF_DIR):
        self.redis_bin_path = redis_bin_path
        self.port = port
        self.max_memory = f"{max_memory}mb"
        self.conf_dir = conf_dir or os.path.expanduser("~/redis_conf")
        self.conf_path = os.path.join(self.conf_dir, "redis.conf")
        self.pid_file = os.path.join(self.conf_dir, "redis.pid")
        self.log_file = os.path.join(self.conf_dir, "redis.log")
        self.redis_pid = None
        print(self.log_file)
        
        atexit.register(self.stop)
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """设置信号处理"""
        def handle_signal(signum, frame):
            self.stop()
            sys.exit(0)
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

    def _create_config(self):
        """创建Redis配置文件"""
        if not _TOOL_CACHE_SWITCH:
            print("❌ 工具缓存开关关闭，Redis服务未启动")
            return
        Path(self.conf_dir).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(self.conf_path):
            print("===== 首次启动，创建Redis配置文件 =====")
            config = f"""bind 127.0.0.1
port {self.port}
maxmemory {self.max_memory}
maxmemory-policy allkeys-lru
protected-mode no
daemonize yes
pidfile {self.pid_file}
logfile {self.log_file}
# save 900 1
# save 300 10
# save 60 10000
save ""
appendonly no
"""
            with open(self.conf_path, 'w', encoding='utf-8') as f:
                f.write(config)

    def start(self):
        """启动Redis"""
        if not _TOOL_CACHE_SWITCH:
            print("❌ 由于环境变量控制，Redis未启动（工具缓存开关关闭）")
            return False
        
        self._create_config()
        self.stop(quiet=True)
        
        redis_server = os.path.join(self.redis_bin_path, "redis-server")
        if not os.path.exists(redis_server):
            raise FileNotFoundError(f"Redis服务程序不存在：{redis_server}")
        
        subprocess.Popen([redis_server, self.conf_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        
        cli = os.path.join(self.redis_bin_path, "redis-cli")
        try:
            res = subprocess.run(
                [cli, "-p", str(self.port), "ping"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,
                text=True
            )
            if "PONG" in res.stdout:
                print(f"✅ Redis {self.port} 启动成功")
                if os.path.exists(self.pid_file):
                    with open(self.pid_file, 'r') as f:
                        self.redis_pid = int(f.read().strip())
                return True
            else:
                raise Exception(f"启动失败：{res.stdout}{res.stderr}")
        except Exception as e:
            print(f"❌ Redis启动失败：{e}")
            print(f"📄 日志路径：{self.log_file}")
            sys.exit(1)

    def stop(self, quiet=False):
        """停止Redis"""
        if not _TOOL_CACHE_SWITCH:
            print("❌ 由于环境变量控制，Redis未启动（工具缓存开关关闭）stop")
            return
        if not quiet:
            print("\n===== 停止Redis =====")
        
        cli = os.path.join(self.redis_bin_path, "redis-cli")
        try:
            subprocess.run([cli, "-p", str(self.port), "shutdown"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            if not quiet:
                print("✅ Redis已主动关闭")
        except:
            if self.redis_pid:
                try:
                    os.kill(self.redis_pid, signal.SIGTERM)
                    time.sleep(2)
                    os.kill(self.redis_pid, 0)  # 检查是否存活
                    os.kill(self.redis_pid, signal.SIGKILL)
                    if not quiet:
                        print("✅ Redis已强制关闭")
                except OSError:
                    if not quiet:
                        print("✅ Redis进程已不存在")
        self.redis_pid = None