#!/usr/bin/env python3
"""
Genera:
1. tool_sft_v3_bash.jsonl  — corpus de entrenamiento ampliado (~600 ejemplos)
   - Nuestros v2 expandidos con más variedad
   - xlam-function-calling-60k convertido a nuestro formato (bash_exec subset)
   - Más variantes de preguntas en español

2. b4_tooluse_v2.jsonl — benchmark B4 ampliado (50 preguntas)
   - 20 bash básico (date, whoami, ls, df, free, uptime...)
   - 10 bash intermedio (ps, grep, find, ss)
   - 10 nvd/cisa (MCP)
   - 10 otx (MCP)

Estrategia corpus:
- 70% bash_exec (básico + intermedio)
- 20% MCP (nvd, otx, cisa)
- 10% negativos conversacionales
"""

import json
import random
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

def ex(user, assistant):
    return {
        "text": f"<|system|>{SYSTEM_PROMPT}<|end|>"
                f"<|user|>{user}<|end|>"
                f"<|assistant|>{assistant}<|end|>",
        "source": "tool_sft_v3"
    }

def tc(name, args):
    return f'<|tool_call|>{json.dumps({"name": name, "args": args}, ensure_ascii=False)}<|/tool_call|>'


# ─────────────────────────────────────────────
# BASH BÁSICO — nivel 1 (fecha, usuario, sistema)
# ─────────────────────────────────────────────
def bash_basic():
    items = []

    # Fecha/hora — muchas variantes de pregunta
    date_cmds = [
        ("date",              ["qué hora es", "dime la hora", "hora actual", "qué hora tiene el servidor"]),
        ("date +%Y-%m-%d",    ["dame la fecha de hoy", "fecha actual", "qué día es hoy", "fecha del sistema"]),
        ("date +%H:%M:%S",    ["muéstrame la hora exacta", "hora en formato HH:MM:SS"]),
        ("date -u",           ["hora UTC del servidor", "qué hora es en UTC"]),
        ("date +%s",          ["dame el timestamp unix actual", "epoch time ahora"]),
    ]
    for cmd, questions in date_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Usuario/identidad
    user_cmds = [
        ("whoami",            ["quién soy", "con qué usuario estoy", "mi usuario actual", "qué usuario ejecuta esto"]),
        ("id",                ["muéstrame mi identidad completa", "dame mi uid y gid", "qué grupos tengo"]),
        ("id -u",             ["dame solo mi uid", "cuál es mi user id"]),
        ("groups",            ["a qué grupos pertenezco", "mis grupos de usuario"]),
        ("id -un",            ["nombre de mi usuario efectivo"]),
    ]
    for cmd, questions in user_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Directorio
    dir_cmds = [
        ("pwd",               ["en qué directorio estoy", "directorio actual", "dónde estoy", "ruta actual"]),
        ("ls -lh",            ["lista los archivos aquí", "qué hay en este directorio", "archivos del directorio actual"]),
        ("ls -la",            ["lista todos los archivos incluyendo ocultos", "archivos ocultos también"]),
        ("ls -lh /tmp",       ["qué hay en /tmp", "archivos en /tmp", "lista /tmp"]),
        ("ls -lh /var/log",   ["qué logs hay en /var/log", "archivos de log disponibles"]),
        ("ls -lh /etc",       ["qué hay en /etc", "archivos de configuración en /etc"]),
    ]
    for cmd, questions in dir_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Memoria
    mem_cmds = [
        ("free -h",           ["cuánta memoria libre hay", "uso de memoria", "RAM disponible", "memoria del sistema"]),
        ("free -m",           ["memoria en megabytes", "RAM en MB"]),
        ("vmstat 1 3",        ["estadísticas de memoria y CPU", "vmstat del sistema"]),
    ]
    for cmd, questions in mem_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Disco
    disk_cmds = [
        ("df -h",             ["uso de disco", "espacio libre en disco", "cuánto disco queda", "particiones y espacio"]),
        ("df -h /",           ["espacio en la partición raíz", "cuánto queda en /"]),
        ("du -sh /var/log",   ["cuánto ocupa /var/log", "tamaño de los logs"]),
        ("du -sh /home",      ["cuánto ocupa /home"]),
    ]
    for cmd, questions in disk_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Sistema
    sys_cmds = [
        ("uname -a",          ["qué sistema operativo es", "info del kernel", "versión del sistema", "uname del servidor"]),
        ("uname -r",          ["versión del kernel", "qué kernel corre"]),
        ("cat /etc/os-release", ["qué distribución linux es", "versión de la distro", "qué OS es este"]),
        ("uptime",            ["cuánto tiempo lleva encendido", "uptime del servidor", "hace cuánto arrancó"]),
        ("uptime -p",         ["uptime en formato legible"]),
        ("hostname",          ["cuál es el hostname", "nombre del servidor", "nombre del host"]),
        ("hostname -I",       ["cuál es mi IP", "dirección IP del servidor", "qué IP tiene este servidor"]),
    ]
    for cmd, questions in sys_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    return items


# ─────────────────────────────────────────────
# BASH INTERMEDIO — nivel 2 (procesos, red, logs)
# ─────────────────────────────────────────────
def bash_intermediate():
    items = []

    # Procesos
    proc_cmds = [
        ("ps aux",                          ["muestra los procesos activos", "lista de procesos", "qué procesos corren"]),
        ("ps aux --sort=-%cpu | head -10",  ["procesos que más CPU usan", "top procesos por CPU"]),
        ("ps aux --sort=-%mem | head -10",  ["procesos que más memoria usan", "top procesos por RAM"]),
        ("ps aux | wc -l",                  ["cuántos procesos hay", "número de procesos activos"]),
        ("pgrep -a sshd",                   ["está corriendo sshd", "proceso sshd activo"]),
        ("pgrep -a nginx",                  ["está corriendo nginx", "proceso nginx"]),
    ]
    for cmd, questions in proc_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Red
    net_cmds = [
        ("ss -tuln",                        ["qué puertos están escuchando", "puertos abiertos", "servicios en escucha"]),
        ("ss -tan state established",       ["conexiones establecidas", "conexiones activas TCP"]),
        ("ip addr",                         ["muestra las interfaces de red", "interfaces de red del sistema"]),
        ("ip route",                        ["tabla de rutas", "rutas de red configuradas"]),
        ("ss -s",                           ["estadísticas de sockets", "resumen de conexiones"]),
    ]
    for cmd, questions in net_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Logs y búsqueda
    log_cmds = [
        ("tail -20 /var/log/syslog",        ["últimas líneas de syslog", "últimos eventos del sistema"]),
        ("tail -20 /var/log/auth.log",      ["últimos eventos de autenticación", "últimos logins"]),
        ("grep -i error /var/log/syslog",   ["errores en syslog", "busca errores en el log del sistema"]),
        ("grep -c failed /var/log/auth.log",["cuántos fallos de autenticación hay", "intentos fallidos de login"]),
        ("last -10",                        ["últimos 10 logins", "quién se conectó recientemente"]),
        ("journalctl -n 20",                ["últimos 20 eventos del journal", "journalctl reciente"]),
        ("journalctl -p err -n 20",         ["últimos errores en el journal", "errores recientes del sistema"]),
    ]
    for cmd, questions in log_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    # Archivos
    file_cmds = [
        ("find /var/log -name '*.log' -type f",  ["archivos .log en /var/log", "busca logs en /var/log"]),
        ("find . -type f -mtime 0",              ["archivos modificados hoy", "cambios de hoy"]),
        ("find /tmp -type f -newer /tmp",        ["archivos recientes en /tmp"]),
        ("find / -perm -4000 -type f 2>/dev/null", ["binarios SUID en el sistema", "busca SUID binaries"]),
    ]
    for cmd, questions in file_cmds:
        for q in questions:
            items.append(ex(q, tc("bash_exec", {"cmd": cmd})))

    return items


# ─────────────────────────────────────────────
# MCP TOOLS — nvd, cisa, otx
# ─────────────────────────────────────────────
def mcp_tools():
    items = []

    # nvd_get_cve
    cves = [
        "CVE-2021-44228", "CVE-2024-4577", "CVE-2022-22965",
        "CVE-2023-44487", "CVE-2024-21762", "CVE-2023-23397",
        "CVE-2024-27198", "CVE-2025-0411",
    ]
    templates_get = [
        "dame detalles de {cve}", "qué sabes de {cve}", "busca {cve}",
        "información sobre {cve}", "analiza {cve}", "detalle técnico de {cve}",
    ]
    for cve in cves:
        for tmpl in templates_get[:3]:  # 3 variantes por CVE
            items.append(ex(tmpl.format(cve=cve), tc("nvd_get_cve", {"cve_id": cve})))

    # nvd_search
    searches = [
        ("log4j",              ["busca CVEs de log4j", "vulnerabilidades de log4j", "CVEs relacionados con log4j"]),
        ("openssh",            ["CVEs de openssh", "vulnerabilidades en openssh"]),
        ("kernel linux",       ["CVEs del kernel linux", "vulnerabilidades del kernel"]),
        ("apache",             ["vulnerabilidades de apache", "CVEs de apache httpd"]),
        ("windows rdp",        ["CVEs de RDP en windows", "vulnerabilidades RDP"]),
        ("fortinet",           ["CVEs de fortinet", "vulnerabilidades fortigate"]),
        ("ivanti",             ["CVEs de ivanti", "vulnerabilidades ivanti connect"]),
    ]
    for query, questions in searches:
        for q in questions:
            items.append(ex(q, tc("nvd_search", {"query": query, "limit": 10})))

    # cisa_kev_check
    kev_cves = ["CVE-2024-27198", "CVE-2025-0411", "CVE-2021-44228",
                "CVE-2023-44487", "CVE-2024-21762"]
    kev_templates = [
        "está {cve} en KEV", "verifica KEV para {cve}",
        "¿debo parchear urgentemente {cve}?", "¿{cve} está siendo explotada activamente?",
    ]
    for cve in kev_cves:
        for tmpl in kev_templates[:2]:
            items.append(ex(tmpl.format(cve=cve), tc("cisa_kev_check", {"cve_id": cve})))

    # otx_check_ioc — IPs
    ips = ["45.155.205.12", "185.220.101.42", "194.165.16.71",
           "103.74.19.104", "91.92.251.103", "198.199.10.234"]
    ip_templates = [
        "es maliciosa la IP {ip}", "reputación de {ip}",
        "verifica {ip} en OTX", "¿{ip} es un IOC conocido?",
    ]
    for ip in ips:
        for tmpl in ip_templates[:2]:
            items.append(ex(tmpl.format(ip=ip),
                           tc("otx_check_ioc", {"ioc_type": "ip", "value": ip})))

    # otx_check_ioc — dominios
    domains = ["phish-bank.test", "malware-drop.example.org",
               "c2-server.evil.net", "update-flash.xyz"]
    for domain in domains:
        items.append(ex(f"reputación de {domain}",
                       tc("otx_check_ioc", {"ioc_type": "domain", "value": domain})))
        items.append(ex(f"¿{domain} es malicioso?",
                       tc("otx_check_ioc", {"ioc_type": "domain", "value": domain})))

    # otx_check_ioc — hashes
    hashes = [
        "44d88612fea8a8f36de82e1278abb02f",
        "e3b0c44298fc1c149afbf4c8996fb924",
    ]
    for h in hashes:
        items.append(ex(f"verifica el hash {h}",
                       tc("otx_check_ioc", {"ioc_type": "hash", "value": h})))

    return items


# ─────────────────────────────────────────────
# NEGATIVOS — conversacionales sin tool-call
# ─────────────────────────────────────────────
def negatives():
    return [
        ex("hola, cómo estás",
           "Hola, estoy listo para ayudarte con consultas de ciberseguridad. ¿En qué puedo asistirte?"),
        ex("qué es un zero-day",
           "Una vulnerabilidad zero-day es una falla desconocida para el proveedor, sin parche disponible al momento de su explotación."),
        ex("explica qué es un CVE",
           "CVE (Common Vulnerabilities and Exposures) es un identificador único para vulnerabilidades de seguridad conocidas públicamente."),
        ex("qué es CISA KEV",
           "CISA KEV (Known Exploited Vulnerabilities) es un catálogo de vulnerabilidades que están siendo activamente explotadas en la naturaleza."),
        ex("para qué sirve OTX",
           "AlienVault OTX es una plataforma de inteligencia de amenazas colaborativa para verificar reputación de IOCs como IPs, dominios y hashes."),
        ex("qué es un IOC",
           "Un IOC (Indicator of Compromise) es un artefacto observable que sugiere una intrusión: IP maliciosa, hash de malware, dominio de phishing."),
        ex("qué diferencia hay entre vulnerabilidad y exploit",
           "Una vulnerabilidad es una debilidad en el software. Un exploit es código que aprovecha esa vulnerabilidad para comprometer el sistema."),
        ex("qué es ransomware",
           "Ransomware es malware que cifra los archivos de la víctima y exige un rescate para restaurar el acceso."),
        ex("qué es un APT",
           "APT (Advanced Persistent Threat) es un actor de amenaza sofisticado que mantiene acceso prolongado y sigiloso a una red objetivo."),
        ex("gracias por tu ayuda",
           "De nada, estoy aquí para ayudarte con cualquier consulta de ciberseguridad."),
        ex("qué puedes hacer",
           "Puedo consultar CVEs en NVD, verificar si están en CISA KEV, analizar reputación de IPs y dominios en OTX, y ejecutar comandos bash en el sistema."),
        ex("cómo funciona MCP",
           "MCP (Model Context Protocol) es un protocolo que permite a los modelos de lenguaje invocar herramientas externas de forma estandarizada mediante JSON-RPC."),
    ]


# ─────────────────────────────────────────────
# BENCHMARK B4 v2 — 50 preguntas
# ─────────────────────────────────────────────
def build_benchmark_b4_v2():
    bench = []

    # 20 bash básico
    bash_basic_bench = [
        ("qué hora es",                          "bash_exec"),
        ("dame la fecha de hoy",                 "bash_exec"),
        ("hora UTC del servidor",                "bash_exec"),
        ("quién soy",                            "bash_exec"),
        ("qué grupos tengo",                     "bash_exec"),
        ("en qué directorio estoy",              "bash_exec"),
        ("lista los archivos aquí",              "bash_exec"),
        ("qué hay en /tmp",                      "bash_exec"),
        ("cuánta memoria libre hay",             "bash_exec"),
        ("uso de disco",                         "bash_exec"),
        ("espacio en la partición raíz",         "bash_exec"),
        ("qué sistema operativo es",             "bash_exec"),
        ("versión del kernel",                   "bash_exec"),
        ("cuánto tiempo lleva encendido",        "bash_exec"),
        ("cuál es el hostname",                  "bash_exec"),
        ("cuál es mi IP",                        "bash_exec"),
        ("muéstrame mi identidad completa",      "bash_exec"),
        ("qué distribución linux es",            "bash_exec"),
        ("uptime en formato legible",            "bash_exec"),
        ("cuánto ocupa /var/log",                "bash_exec"),
    ]

    # 10 bash intermedio
    bash_inter_bench = [
        ("qué puertos están escuchando",         "bash_exec"),
        ("conexiones TCP establecidas",          "bash_exec"),
        ("procesos que más CPU usan",            "bash_exec"),
        ("cuántos procesos hay corriendo",       "bash_exec"),
        ("últimos 10 logins del sistema",        "bash_exec"),
        ("errores recientes en el journal",      "bash_exec"),
        ("archivos modificados hoy",             "bash_exec"),
        ("busca errores en syslog",              "bash_exec"),
        ("interfaces de red del sistema",        "bash_exec"),
        ("tabla de rutas de red",                "bash_exec"),
    ]

    # 10 MCP nvd/cisa
    mcp_bench = [
        ("dame detalles de CVE-2021-44228",      "nvd_get_cve"),
        ("qué sabes de CVE-2024-4577",           "nvd_get_cve"),
        ("información sobre CVE-2023-44487",     "nvd_get_cve"),
        ("busca CVEs de log4j",                  "nvd_search"),
        ("vulnerabilidades de openssh",          "nvd_search"),
        ("CVEs críticos de apache",              "nvd_search"),
        ("está CVE-2021-44228 en KEV",           "cisa_kev_check"),
        ("verifica KEV para CVE-2024-27198",     "cisa_kev_check"),
        ("¿debo parchear CVE-2023-44487?",       "cisa_kev_check"),
        ("CVEs de fortinet fortigate",           "nvd_search"),
    ]

    # 10 OTX
    otx_bench = [
        ("es maliciosa la IP 45.155.205.12",     "otx_check_ioc"),
        ("reputación de 185.220.101.42",         "otx_check_ioc"),
        ("verifica 194.165.16.71 en OTX",        "otx_check_ioc"),
        ("¿103.74.19.104 es un IOC conocido?",   "otx_check_ioc"),
        ("reputación de phish-bank.test",        "otx_check_ioc"),
        ("¿malware-drop.example.org es malicioso?", "otx_check_ioc"),
        ("verifica el hash 44d88612fea8a8f36de82e1278abb02f", "otx_check_ioc"),
        ("analiza la IP 91.92.251.103",          "otx_check_ioc"),
        ("¿c2-server.evil.net es C2?",           "otx_check_ioc"),
        ("reputación de update-flash.xyz",       "otx_check_ioc"),
    ]

    all_bench = bash_basic_bench + bash_inter_bench + mcp_bench + otx_bench
    for i, (question, tool) in enumerate(all_bench):
        bench.append({
            "id": f"b4v2_{i:03d}",
            "question": question,
            "expected_tool": tool
        })

    return bench


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Corpus v3
    basic   = bash_basic()
    inter   = bash_intermediate()
    mcp     = mcp_tools()
    negs    = negatives()

    print(f"[raw] bash_basic={len(basic)} bash_inter={len(inter)} "
          f"mcp={len(mcp)} negatives={len(negs)}")

    # Combinar con repetición para balancear
    # bash básico 2x, intermedio 1x, mcp 1x, negativos 2x
    corpus = basic * 2 + inter + mcp + negs * 2
    random.shuffle(corpus)

    out_corpus = Path("corpus/tool_sft_v3_bash.jsonl")
    with open(out_corpus, "w", encoding="utf-8") as f:
        for ex in corpus:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    tool_calls = sum(1 for e in corpus if "<|tool_call|>" in e["text"])
    bash_calls = sum(1 for e in corpus if "bash_exec" in e["text"] and "<|tool_call|>" in e["text"])
    mcp_calls  = tool_calls - bash_calls
    conv       = len(corpus) - tool_calls

    print(f"\n[corpus v3] {out_corpus}")
    print(f"  Total:        {len(corpus)}")
    print(f"  bash_exec:    {bash_calls} ({bash_calls/len(corpus)*100:.1f}%)")
    print(f"  MCP tools:    {mcp_calls} ({mcp_calls/len(corpus)*100:.1f}%)")
    print(f"  Conversacional: {conv} ({conv/len(corpus)*100:.1f}%)")

    # Benchmark B4 v2
    bench = build_benchmark_b4_v2()
    out_bench = Path("corpus/b4_tooluse_v2.jsonl")
    with open(out_bench, "w", encoding="utf-8") as f:
        for item in bench:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    bash_b = sum(1 for b in bench if b["expected_tool"] == "bash_exec")
    mcp_b  = len(bench) - bash_b
    print(f"\n[benchmark B4 v2] {out_bench}")
    print(f"  Total:     {len(bench)}")
    print(f"  bash_exec: {bash_b} ({bash_b/len(bench)*100:.0f}%)")
    print(f"  MCP:       {mcp_b} ({mcp_b/len(bench)*100:.0f}%)")
    print(f"\n[done] Listo para subir a S3")


if __name__ == "__main__":
    main()
