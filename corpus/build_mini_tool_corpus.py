#!/usr/bin/env python3
"""
Genera corpus tool-use para Mini (Qwen2.5-1.5B + LoRA).

Objetivo: ratio 1:20 tool-use/total → ~3,000 ejemplos tool-use.

Fuentes:
1. Nuestro tool_sft_v3_bash.jsonl (296 ejemplos, ya en formato correcto)
2. Salesforce/xlam-function-calling-60k (filtrado a bash/system tools)
3. glaiveai/glaive-function-calling-v2 (function calling simple, convertido)
4. Sintéticos adicionales generados aquí (bash + MCP en español)

Formato de salida: ChatML con nuestros tokens especiales
<|system|>...<|end|><|user|>...<|end|><|assistant|><|tool_call|>{"name":"...","args":{...}}<|/tool_call|><|end|>
"""

import json
import random
import re
from pathlib import Path

random.seed(42)

SYSTEM_PROMPT = (
    "Eres VectraYX, un asistente de ciberseguridad en español. "
    "Tienes 5 herramientas MCP disponibles:\n"
    "- nvd_get_cve(cve_id): obtener detalle de un CVE\n"
    "- nvd_search(query, limit): buscar CVEs por palabra clave\n"
    "- cisa_kev_check(cve_id): comprobar si un CVE está en el catálogo KEV\n"
    "- otx_check_ioc(ioc_type, value): reputación de IOC (ip, domain, hash)\n"
    "- bash_exec(cmd): ejecutar comando shell local\n"
    "Cuando la pregunta requiera datos externos o ejecutar algo, emite EXACTAMENTE:\n"
    '<|tool_call|>{"name":"<tool>","args":{...}}<|/tool_call|>\n'
    "Si la pregunta es conversacional o conceptual, responde en prosa SIN llamar herramientas."
)

def make_ex(user, assistant, source="synthetic"):
    return {
        "text": (f"<|system|>{SYSTEM_PROMPT}<|end|>"
                 f"<|user|>{user}<|end|>"
                 f"<|assistant|>{assistant}<|end|>"),
        "source": source
    }

def tc(name, args):
    return f'<|tool_call|>{json.dumps({"name": name, "args": args}, ensure_ascii=False)}<|/tool_call|>'


# ─── 1. Cargar nuestro corpus v3 existente ───────────────────────────────────
def load_v3():
    path = Path("corpus/tool_sft_v3_bash.jsonl")
    if not path.exists():
        print("[warn] tool_sft_v3_bash.jsonl no encontrado")
        return []
    items = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    print(f"[v3] {len(items)} ejemplos cargados")
    return items


# ─── 2. Convertir xlam-function-calling-60k ──────────────────────────────────
def convert_xlam(max_examples=1000):
    """
    Descarga y convierte xlam-function-calling-60k.
    Filtra solo ejemplos con herramientas tipo bash/system/file.
    """
    try:
        from datasets import load_dataset
        print("[xlam] Descargando dataset...")
        ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train",
                          trust_remote_code=True)
        print(f"[xlam] {len(ds)} ejemplos totales")
    except Exception as e:
        print(f"[xlam] Error descargando: {e}")
        return []

    # Keywords que indican herramientas bash/system
    bash_keywords = {
        "execute", "run", "shell", "bash", "command", "cmd", "terminal",
        "file", "directory", "path", "process", "system", "os", "disk",
        "memory", "cpu", "network", "port", "socket", "ping", "curl",
        "wget", "grep", "find", "ls", "cat", "echo", "date", "time",
        "whoami", "hostname", "uname", "ps", "top", "df", "du", "free"
    }

    converted = []
    for item in ds:
        try:
            query = item.get("query", "")
            answers = item.get("answers", "[]")
            tools = item.get("tools", "[]")

            if isinstance(answers, str):
                answers = json.loads(answers)
            if isinstance(tools, str):
                tools = json.loads(tools)

            if not answers or not isinstance(answers, list):
                continue

            # Filtrar por herramientas bash-like
            tool_names = [str(t.get("name", "")).lower() for t in tools
                          if isinstance(t, dict)]
            answer_names = [str(a.get("name", "")).lower() for a in answers
                            if isinstance(a, dict)]
            all_names = " ".join(tool_names + answer_names)

            if not any(kw in all_names for kw in bash_keywords):
                continue

            # Convertir primera respuesta a nuestro formato
            ans = answers[0]
            if not isinstance(ans, dict):
                continue

            tool_name = ans.get("name", "")
            tool_args = ans.get("arguments", ans.get("args", {}))
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except Exception:
                    continue

            # Mapear a bash_exec si es una herramienta de sistema
            if any(kw in tool_name.lower() for kw in bash_keywords):
                # Construir comando bash desde los argumentos
                cmd_parts = []
                for k, v in tool_args.items():
                    if isinstance(v, str) and v:
                        cmd_parts.append(str(v))
                cmd = " ".join(cmd_parts) if cmd_parts else tool_name
                assistant_text = tc("bash_exec", {"cmd": cmd})
            else:
                # Mantener como tool-call genérico
                assistant_text = tc(tool_name, tool_args)

            converted.append(make_ex(query, assistant_text, source="xlam"))

            if len(converted) >= max_examples:
                break

        except Exception:
            continue

    print(f"[xlam] {len(converted)} ejemplos convertidos")
    return converted


# ─── 3. Convertir glaive-function-calling-v2 ─────────────────────────────────
def convert_glaive(max_examples=1000):
    """
    Convierte glaive-function-calling-v2.
    Filtra ejemplos simples (single tool call, no multi-turn complejo).
    """
    try:
        from datasets import load_dataset
        print("[glaive] Descargando dataset...")
        ds = load_dataset("glaiveai/glaive-function-calling-v2", split="train",
                          trust_remote_code=True)
        print(f"[glaive] {len(ds)} ejemplos totales")
    except Exception as e:
        print(f"[glaive] Error descargando: {e}")
        return []

    converted = []
    for item in ds:
        try:
            chat = item.get("chat", "")
            system = item.get("system", "")

            # Buscar pattern: USER: ... ASSISTANT: <functioncall> {...}
            user_match = re.search(r'USER:\s*(.+?)(?=ASSISTANT:|$)', chat,
                                   re.DOTALL)
            func_match = re.search(
                r'ASSISTANT:\s*<functioncall>\s*(\{.+?\})', chat, re.DOTALL)

            if not user_match or not func_match:
                continue

            user_text = user_match.group(1).strip()
            func_json_str = func_match.group(1).strip()

            try:
                func_data = json.loads(func_json_str)
            except Exception:
                continue

            func_name = func_data.get("name", "")
            func_args = func_data.get("arguments", {})
            if isinstance(func_args, str):
                try:
                    func_args = json.loads(func_args)
                except Exception:
                    func_args = {"input": func_args}

            # Solo herramientas simples (no multi-step)
            if not func_name or len(user_text) > 500:
                continue

            assistant_text = tc(func_name, func_args)
            converted.append(make_ex(user_text, assistant_text, source="glaive"))

            if len(converted) >= max_examples:
                break

        except Exception:
            continue

    print(f"[glaive] {len(converted)} ejemplos convertidos")
    return converted


# ─── 4. Sintéticos adicionales en español ────────────────────────────────────
def generate_synthetic_spanish(n=500):
    """Genera ejemplos sintéticos adicionales en español con variedad."""
    items = []

    # Bash básico — más variantes
    bash_pairs = [
        # Fecha/hora
        ("¿qué hora tiene el servidor?", tc("bash_exec", {"cmd": "date"})),
        ("necesito saber la fecha exacta", tc("bash_exec", {"cmd": "date +%Y-%m-%d"})),
        ("dame el timestamp unix", tc("bash_exec", {"cmd": "date +%s"})),
        ("hora en formato ISO", tc("bash_exec", {"cmd": "date -Iseconds"})),
        ("¿cuándo fue el último reinicio?", tc("bash_exec", {"cmd": "who -b"})),
        # Sistema
        ("¿qué versión de linux corre?", tc("bash_exec", {"cmd": "cat /etc/os-release"})),
        ("muéstrame el kernel", tc("bash_exec", {"cmd": "uname -r"})),
        ("arquitectura del sistema", tc("bash_exec", {"cmd": "uname -m"})),
        ("hostname del servidor", tc("bash_exec", {"cmd": "hostname -f"})),
        ("variables de entorno", tc("bash_exec", {"cmd": "env | sort"})),
        # Memoria/disco
        ("¿cuánta RAM tiene el servidor?", tc("bash_exec", {"cmd": "free -h"})),
        ("uso de swap", tc("bash_exec", {"cmd": "free -h | grep Swap"})),
        ("espacio en /var", tc("bash_exec", {"cmd": "df -h /var"})),
        ("directorios más grandes", tc("bash_exec", {"cmd": "du -sh /* 2>/dev/null | sort -rh | head -10"})),
        ("inodos disponibles", tc("bash_exec", {"cmd": "df -i"})),
        # Procesos
        ("¿qué proceso usa el puerto 80?", tc("bash_exec", {"cmd": "ss -tlnp | grep :80"})),
        ("procesos del usuario root", tc("bash_exec", {"cmd": "ps aux | grep root"})),
        ("árbol de procesos", tc("bash_exec", {"cmd": "pstree -p"})),
        ("procesos zombie", tc("bash_exec", {"cmd": "ps aux | grep Z"})),
        ("tiempo de CPU por proceso", tc("bash_exec", {"cmd": "ps aux --sort=-%cpu | head -20"})),
        # Red
        ("conexiones al puerto 443", tc("bash_exec", {"cmd": "ss -tn | grep :443"})),
        ("tabla ARP", tc("bash_exec", {"cmd": "arp -n"})),
        ("rutas de red", tc("bash_exec", {"cmd": "ip route show"})),
        ("estadísticas de red", tc("bash_exec", {"cmd": "netstat -s 2>/dev/null || ss -s"})),
        ("interfaces activas", tc("bash_exec", {"cmd": "ip link show up"})),
        # Seguridad
        ("usuarios con UID 0", tc("bash_exec", {"cmd": "awk -F: '$3==0' /etc/passwd"})),
        ("últimos comandos ejecutados", tc("bash_exec", {"cmd": "history | tail -20"})),
        ("archivos con permisos 777", tc("bash_exec", {"cmd": "find / -perm 777 -type f 2>/dev/null | head -20"})),
        ("crontabs del sistema", tc("bash_exec", {"cmd": "crontab -l 2>/dev/null; ls /etc/cron*"})),
        ("servicios activos", tc("bash_exec", {"cmd": "systemctl list-units --type=service --state=running"})),
        # Logs
        ("errores de autenticación hoy", tc("bash_exec", {"cmd": "grep 'Failed password' /var/log/auth.log | tail -20"})),
        ("últimas entradas de syslog", tc("bash_exec", {"cmd": "tail -50 /var/log/syslog"})),
        ("logs del kernel", tc("bash_exec", {"cmd": "dmesg | tail -30"})),
        ("intentos de login fallidos", tc("bash_exec", {"cmd": "faillog -a | head -20"})),
        ("logs de sudo", tc("bash_exec", {"cmd": "grep sudo /var/log/auth.log | tail -20"})),
    ]

    # MCP tools — más variantes
    mcp_pairs = [
        # NVD
        ("¿hay CVEs críticos de nginx?", tc("nvd_search", {"query": "nginx", "limit": 10})),
        ("vulnerabilidades de wordpress", tc("nvd_search", {"query": "wordpress", "limit": 10})),
        ("CVEs de mysql 8.0", tc("nvd_search", {"query": "mysql 8.0", "limit": 10})),
        ("busca CVEs de docker", tc("nvd_search", {"query": "docker", "limit": 10})),
        ("vulnerabilidades de python", tc("nvd_search", {"query": "python", "limit": 10})),
        ("CVEs de kubernetes", tc("nvd_search", {"query": "kubernetes", "limit": 10})),
        ("vulnerabilidades de jenkins", tc("nvd_search", {"query": "jenkins", "limit": 10})),
        ("CVEs de gitlab", tc("nvd_search", {"query": "gitlab", "limit": 10})),
        # NVD get
        ("detalle de CVE-2023-0386", tc("nvd_get_cve", {"cve_id": "CVE-2023-0386"})),
        ("información de CVE-2024-3094", tc("nvd_get_cve", {"cve_id": "CVE-2024-3094"})),
        ("CVE-2023-44487 detalles", tc("nvd_get_cve", {"cve_id": "CVE-2023-44487"})),
        ("qué es CVE-2024-6387", tc("nvd_get_cve", {"cve_id": "CVE-2024-6387"})),
        # KEV
        ("¿CVE-2023-0386 está en KEV?", tc("cisa_kev_check", {"cve_id": "CVE-2023-0386"})),
        ("verifica KEV para CVE-2024-3094", tc("cisa_kev_check", {"cve_id": "CVE-2024-3094"})),
        ("¿se explota activamente CVE-2024-6387?", tc("cisa_kev_check", {"cve_id": "CVE-2024-6387"})),
        # OTX
        ("¿es maliciosa 91.92.251.103?", tc("otx_check_ioc", {"ioc_type": "ip", "value": "91.92.251.103"})),
        ("reputación de 198.199.10.234", tc("otx_check_ioc", {"ioc_type": "ip", "value": "198.199.10.234"})),
        ("¿evil-update.xyz es phishing?", tc("otx_check_ioc", {"ioc_type": "domain", "value": "evil-update.xyz"})),
        ("verifica hash e3b0c44298fc1c149afbf4c8996fb924", tc("otx_check_ioc", {"ioc_type": "hash", "value": "e3b0c44298fc1c149afbf4c8996fb924"})),
    ]

    # Conversacionales
    conv_pairs = [
        ("¿qué es CVSS?", "CVSS (Common Vulnerability Scoring System) es un estándar para calificar la severidad de vulnerabilidades de seguridad en una escala de 0 a 10."),
        ("explica qué es un buffer overflow", "Un buffer overflow ocurre cuando un programa escribe más datos en un buffer de los que puede contener, sobrescribiendo memoria adyacente y potencialmente permitiendo ejecución de código arbitrario."),
        ("¿qué es MITRE ATT&CK?", "MITRE ATT&CK es un framework de conocimiento sobre tácticas y técnicas de adversarios basado en observaciones del mundo real, usado para modelar amenazas y mejorar defensas."),
        ("¿qué diferencia hay entre IDS e IPS?", "Un IDS (Intrusion Detection System) detecta y alerta sobre actividad sospechosa. Un IPS (Intrusion Prevention System) además bloquea activamente el tráfico malicioso."),
        ("¿qué es un CVE?", "CVE (Common Vulnerabilities and Exposures) es un identificador único para vulnerabilidades de seguridad conocidas públicamente, mantenido por MITRE."),
        ("¿qué es phishing?", "Phishing es un ataque de ingeniería social donde el atacante se hace pasar por una entidad confiable para robar credenciales u otra información sensible."),
        ("¿qué es un APT?", "APT (Advanced Persistent Threat) es un actor de amenaza sofisticado que mantiene acceso prolongado y sigiloso a una red objetivo, generalmente con motivación geopolítica o económica."),
        ("¿qué es OWASP?", "OWASP (Open Web Application Security Project) es una fundación sin fines de lucro que publica recursos sobre seguridad de aplicaciones web, incluyendo el famoso OWASP Top 10."),
        ("¿qué es un zero-day?", "Una vulnerabilidad zero-day es una falla desconocida para el proveedor del software, sin parche disponible al momento de su explotación."),
        ("¿qué es ransomware?", "Ransomware es malware que cifra los archivos de la víctima y exige un rescate económico para restaurar el acceso."),
    ]

    all_pairs = bash_pairs + mcp_pairs
    conv_items = [make_ex(u, a, "synthetic_conv") for u, a in conv_pairs]

    # Repetir para llegar a n ejemplos
    tool_items = [make_ex(u, a, "synthetic_tool") for u, a in all_pairs]
    while len(tool_items) < n - len(conv_items):
        tool_items.extend([make_ex(u, a, "synthetic_tool") for u, a in all_pairs])

    items = tool_items[:n - len(conv_items)] + conv_items
    random.shuffle(items)
    print(f"[synthetic] {len(items)} ejemplos generados")
    return items


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    all_examples = []

    # 1. Nuestro corpus v3 (ya en formato correcto)
    v3 = load_v3()
    all_examples.extend(v3)

    # 2. Sintéticos adicionales en español
    synthetic = generate_synthetic_spanish(n=500)
    all_examples.extend(synthetic)

    # 3. Intentar descargar datasets externos
    xlam = convert_xlam(max_examples=1000)
    all_examples.extend(xlam)

    glaive = convert_glaive(max_examples=1000)
    all_examples.extend(glaive)

    # Si no se pudieron descargar los externos, generar más sintéticos
    if len(xlam) == 0 and len(glaive) < 100:
        print("[fallback] Generando más sintéticos para compensar...")
        extra = generate_synthetic_spanish(n=2000)
        all_examples.extend(extra)
    elif len(all_examples) < 2500:
        print("[boost] Generando sintéticos adicionales para llegar a 3000...")
        extra = generate_synthetic_spanish(n=2500 - len(all_examples))
        all_examples.extend(extra)

    # Shuffle final
    random.shuffle(all_examples)

    # Stats
    tool_calls = sum(1 for e in all_examples if "<|tool_call|>" in e["text"])
    conv = len(all_examples) - tool_calls
    print(f"\n[CORPUS MINI STATS]")
    print(f"  Total:          {len(all_examples)}")
    print(f"  Tool-calls:     {tool_calls} ({tool_calls/len(all_examples)*100:.1f}%)")
    print(f"  Conversacional: {conv} ({conv/len(all_examples)*100:.1f}%)")
    print(f"  Ratio vs 60K SFT: 1:{60000//tool_calls}")

    # Guardar
    out = Path("corpus/tool_sft_mini_v1.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
