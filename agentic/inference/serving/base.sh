local_ip=`hostname -I | cut -d' ' -f1`
nic_name=$(ifconfig | grep -B 1 "$local_ip" | head -n 1 | awk '{print $1}' | sed 's/://')

export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$nic_name}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$nic_name}
if [ "${NPU_NUM}" = "" ]; then
  NPU_NUM=`npu-smi info | grep -o '910B\|Ascend910' | wc -l`
  if [ "${NPU_NUM}" = 8 ]; then
    # For A2
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  elif [ "${NPU_NUM}" = 16 ]; then
    # For A3
    export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  fi
fi

export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=fork
export VLLM_ENABLE_MC2=0
export USING_LCCL_COM=0
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:=1200}

# For simplifying writing the additional config
json_compact_or_die() {
  local src="$1"
  local py="$(command -v python3 || command -v python)"
  if [ -z "$py" ]; then
    echo "[FATAL] python3 未安装，无法校验 JSON" >&2
    exit 1
  fi

  local tmp=""
  if [ "$src" = "-" ]; then
    tmp="$(mktemp)"
    cat >"$tmp"
    src="$tmp"
  fi

  "$py" - "$src" <<'PY'
import sys, json, pathlib
p = pathlib.Path(sys.argv[1])
try:
    txt = p.read_text(encoding="utf-8")
    obj = json.loads(txt)
except Exception as e:
    print(f"[FATAL] JSON 语法错误: {e}", file=sys.stderr)
    sys.exit(2)
print(json.dumps(obj, separators=(",",":")))
PY
  local rc=$?
  [ -n "$tmp" ] && rm -f "$tmp"
  return $rc
}
