from agents.mcp import create_static_tool_filter, MCPServer
import random
from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass, field
import heapq
import threading

@dataclass
class ServerState:
    # host: str
    # port: int
    # url: str
    tools: Set[str] = field(default_factory=set)  # 初始化时记录一次

    inflight: int = 0                 # 该 server 当前进行中的调用数（本客户端视角）
    last_ok_ts: float = 0.0           # 最近一次成功时间（可选）
    consec_failures: int = 0          # 连续失败次数（用于退避）
    penalty_until_ts: float = 0.0     # 熔断惩罚窗口（epoch 秒）

    def score(self) -> float:
        """越小越好：并发越少分越小；熔断期直接加大分。"""
        #penalty = 1000.0 if now < self.penalty_until_ts else 0.0
        #return float(self.inflight) + penalty
        return float(self.inflight)


class ServerManager:
    def __init__(self):
        # 初始化：{ "ip:port": 0 }
        self.servers = {}

    def add_server(self, server_id):
        self.servers[server_id] = 0

    def select_server(self):
        # 1. 找到当前的最小并发数
        min_concurrency = min(self.servers.values())

        # 2. 找出所有并发数等于最小值的服务器（处理并发一致的情况）
        candidates = [addr for addr, count in self.servers.items()
                      if count == min_concurrency]

        # 3. 随机选择一个并增加计数
        selected = random.choice(candidates)
        self.servers[selected] += 1

        return selected

    def release_server(self, addr):
        """任务完成后，手动将并发次数减 1 (或根据你的需求调整)"""
        if addr in self.servers and self.servers[addr] > 0:
            self.servers[addr] -= 1

class MCPClient():
    def __init__(self, server: MCPServer, name):
        self.server = server
        self.state: ServerState = ServerState()
        self.name = name

    def __lt__(self, other):
        if isinstance(other, MCPClient):
            return self.name < other.name
        return NotImplemented


class LoadBalancer:
    def __init__(self, servers: Optional[list] = None):
        """
        初始化负载均衡器
        :param servers: 初始服务器列表（可选）
        """
        self.state_map: Dict[Any, float] = {}  # server -> current state value
        self.heap = []  # min-heap of (state_value, server)

        if servers:
            for server in servers:
                self.add_server(server, initial_state=0.0)

    def add_server(self, server: Any, initial_state: float = 0.0) -> None:
        """添加新服务器，并设置初始状态值"""
        if server in self.state_map:
            return  # 或 raise ValueError("Server already exists")
        self.state_map[server] = initial_state
        heapq.heappush(self.heap, (initial_state, server))

    def update_state(self, server: Any, new_state: float) -> bool:
        """
        外部主动更新服务器的状态值
        :return: 是否成功更新（服务器存在则成功）
        """
        if server not in self.state_map:
            return False
        self.state_map[server] = new_state
        # 将新状态推入堆（旧状态后续通过懒删除清理）
        heapq.heappush(self.heap, (new_state, server))
        return True

    def get_least_state_server(self) -> Any:
        """
        返回当前状态值最小的服务器（不修改状态！）
        """
        if not self.heap:
            raise RuntimeError("No servers available")

        # 懒删除：跳过所有过期堆顶
        while self.heap:
            state_val, server = self.heap[0]
            current_state = self.state_map.get(server, None)
            # 如果服务器已被移除，或堆中状态 ≠ 当前状态 → 过期
            if current_state is None or current_state != state_val:
                heapq.heappop(self.heap)
            else:
                break

        if not self.heap:
            raise RuntimeError("No valid servers found")

        # 注意：这里只返回服务器，不 pop 堆！因为状态未变
        _, server = self.heap[0]
        return server

    def remove_server(self, server: Any) -> bool:
        """移除服务器（状态不再维护）"""
        if server in self.state_map:
            del self.state_map[server]
            return True
        return False

    def get_states(self) -> dict:
        """返回当前所有服务器的状态（用于调试）"""
        return self.state_map.copy()


class McpHubServer():

    def __init__(self, agent_name):
        # To do: Traj pred
        self.agent_name = agent_name
        self.mcp_client_list: list(MCPClient) = []
        self.server_tool_list = []
        self.load_balancer: LoadBalancer = None
        self.current_index = 0
        self.tool_set_cache = {}
        self.server_to_servername = {}
        self.servername_to_server = {}

    def dispatch_roundrobin(self):
        if self.mcp_client_list:
            server = self.mcp_client_list[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.mcp_client_list)
            return [server.server]
        else:
            return [random.choice(self.mcp_client_list).server]

    def dispatch_random(self):
        return [random.choice(self.mcp_client_list).server]

    def dispatch(self):
        if self.mcp_client_list:
            server = self.load_balancer.get_least_state_server()
            server.state.inflight += 1
            self.load_balancer.update_state(server, server.state.score())
            return [self.load_balancer.get_least_state_server().server]
        else:
            return [random.choice(self.mcp_client_list).server]

class MCP_Hub():
    def __init__(self):
        self.stateless_server = {}
        self.mcp_server_manager = ServerManager()
        self.mcp_call_lock = threading.Lock()
