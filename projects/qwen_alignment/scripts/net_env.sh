#!/usr/bin/env bash
# Shared terminal network setup for dataset/model downloads.
# Source this file from project scripts; do not execute it as a standalone tool.

if [[ -z "${PROJECT_DIR:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

export HF_HOME="${HF_HOME:-$PROJECT_DIR/.hf_home}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
_user_hf_endpoint="${HF_ENDPOINT:-}"

_proxy_host="${LOCAL_PROXY_HOST:-127.0.0.1}"
_http_proxy_port="${LOCAL_HTTP_PROXY_PORT:-15236}"
_socks_proxy_port="${LOCAL_SOCKS_PROXY_PORT:-15235}"

_port_open() {
  local host="$1"
  local port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -z -G 1 "$host" "$port" >/dev/null 2>&1
  else
    return 1
  fi
}

if [[ "${LLM_COURSE_DISABLE_PROXY:-0}" != "1" ]]; then
  if [[ "${LLM_COURSE_FORCE_PROXY:-0}" == "1" ]] || _port_open "$_proxy_host" "$_http_proxy_port"; then
    export HTTP_PROXY="${HTTP_PROXY:-http://$_proxy_host:$_http_proxy_port}"
    export HTTPS_PROXY="${HTTPS_PROXY:-http://$_proxy_host:$_http_proxy_port}"
    export http_proxy="${http_proxy:-$HTTP_PROXY}"
    export https_proxy="${https_proxy:-$HTTPS_PROXY}"
  fi

  if [[ "${LLM_COURSE_ENABLE_SOCKS_PROXY:-0}" == "1" ]] &&
    { [[ "${LLM_COURSE_FORCE_PROXY:-0}" == "1" ]] || _port_open "$_proxy_host" "$_socks_proxy_port"; }; then
    export ALL_PROXY="${ALL_PROXY:-socks5h://$_proxy_host:$_socks_proxy_port}"
    export all_proxy="${all_proxy:-$ALL_PROXY}"
  fi
fi

if [[ -z "$_user_hf_endpoint" ]]; then
  if [[ -n "${HTTPS_PROXY:-${https_proxy:-}}" ]]; then
    export HF_ENDPOINT="https://huggingface.co"
  else
    export HF_ENDPOINT="https://hf-mirror.com"
  fi
else
  export HF_ENDPOINT="$_user_hf_endpoint"
fi

export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,::1}"
export no_proxy="${no_proxy:-$NO_PROXY}"
