#!/usr/bin/env python3
"""
Generate a focused tool-use SFT corpus for VectraYX Nano/Base.

Output schema (JSONL, one example per line):
    {"messages": [
        {"role": "system", "content": "<system prompt>"},
        {"role": "user",   "content": "<prompt>"},
        {"role": "assistant", "content": "<|tool_call|>{...}<|/tool_call|>"}
    ]}

For 'negative' examples (no tool needed), assistant emits prose without <|tool_call|>.

Targets:
    --size 1500   → Fase 1 canario
    --size 8000   → Fase 2 (con --curriculum)
"""
import argparse, json, random, hashlib
from pathlib import Path

SYSTEM_PROMPT = (
    "Eres VectraYX, un asistente de ciberseguridad en español. "
    "Tienes 5 herramientas MCP disponibles:\n"
    "- nvd_get_cve(cve_id): obtener detalle de un CVE\n"
    "- nvd_search(query, limit): buscar CVEs por palabra clave\n"
    "- cisa_kev_check(cve_id): comprobar si un CVE está en el catálogo KEV\n"
    "- otx_check_ioc(ioc_type, value): reputación de IOC (ip, domain, hash)\n"
    "- bash_exec(cmd): ejecutar comando shell local\n"
    "Cuando la pregunta requiera datos externos, emite EXACTAMENTE:\n"
    "<|tool_call|>{\"name\":\"<tool>\",\"args\":{...}}<|/tool_call|>\n"
    "Si la pregunta es conversacional o conceptual, responde en prosa SIN llamar herramientas."
)

# CVE pool (real IDs from 2024-2026, varied)
CVES = [
    "CVE-2024-3094", "CVE-2024-4577", "CVE-2024-21762", "CVE-2024-23897",
    "CVE-2024-1709", "CVE-2024-21413", "CVE-2024-3400", "CVE-2024-27198",
    "CVE-2025-21333", "CVE-2025-0411", "CVE-2025-24813", "CVE-2025-0282",
    "CVE-2023-44487", "CVE-2023-23397", "CVE-2023-4863", "CVE-2023-22515",
    "CVE-2021-44228", "CVE-2022-22965", "CVE-2022-30190", "CVE-2021-34527",
]

# Search queries
SEARCH_QUERIES = [
    "log4j", "apache struts", "spring4shell", "follina", "printnightmare",
    "exchange proxylogon", "fortinet ssl-vpn", "ivanti connect secure",
    "moveit transfer", "citrix bleed", "openssh regresshion", "ssh",
    "RCE jenkins", "deserialización java", "xss reflejado wordpress",
    "kubernetes privilege escalation", "docker escape", "kernel linux race",
]

# IOC samples (synthetic, no real malicious data)
IPS    = ["185.220.101.42", "45.155.205.12", "194.165.16.71", "103.74.19.104"]
DOMAINS = ["evil-c2.example", "phish-bank.test", "malware-drop.example.org"]
HASHES = [
    "44d88612fea8a8f36de82e1278abb02f",  # eicar
    "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0f",
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
]

# Bash tasks
BASH_TASKS = [
    ("listar puertos abiertos en el host local", "ss -tlnp"),
    ("ver procesos consumiendo más CPU", "ps aux --sort=-%cpu | head -5"),
    ("revisar el log de autenticación", "tail -50 /var/log/auth.log"),
    ("ver conexiones de red establecidas", "ss -tan state established"),
    ("listar archivos modificados en las últimas 24h en /etc", "find /etc -mtime -1"),
    ("buscar SUID binarios", "find / -perm -4000 -type f 2>/dev/null"),
]

# Templates de fraseo (single-tool)
PHRASE_TEMPLATES = {
    "nvd_get_cve": [
        "dame info de {cve}", "qué sabes de {cve}", "info sobre {cve}",
        "consulta {cve}", "busca el CVE {cve}", "{cve} qué es",
        "quiero detalles de {cve}", "trae {cve} de NVD",
        "podrías revisar {cve}?", "ese {cve} qué tan grave es",
    ],
    "nvd_search": [
        "busca CVEs de {q}", "qué CVEs hay de {q}", "vulnerabilidades en {q}",
        "consulta NVD sobre {q}", "lista los CVEs relacionados con {q}",
        "ataques recientes de {q}", "qué encontraste de {q} en NVD",
    ],
    "cisa_kev_check": [
        "{cve} está en KEV?", "comprueba si {cve} está siendo explotado activamente",
        "el catálogo KEV tiene {cve}?", "verifica KEV para {cve}",
        "está {cve} en el listado de CISA?",
    ],
    "otx_check_ioc": [
        "reputación de {value}", "qué dice OTX de {value}",
        "es maliciosa la {ioc_type} {value}?", "verifica {value} en OTX",
        "chequea la reputación de {value}",
    ],
    "bash_exec": [
        "ejecuta: {desc}", "corre el comando para {desc}",
        "necesito {desc}", "haz un script bash para {desc}",
        "puedes {desc}?", "{desc}, por favor",
    ],
}

# Negativos: preguntas que NO requieren tool
NEGATIVE_PROMPTS = [
    ("qué es ransomware?", "El ransomware es un tipo de malware que cifra archivos del sistema y exige un pago para restaurar el acceso."),
    ("explícame el modelo CIA", "El modelo CIA describe los tres pilares de la seguridad de la información: Confidencialidad, Integridad y Disponibilidad."),
    ("hola, cómo estás?", "Hola, estoy listo para ayudarte con consultas de ciberseguridad. ¿En qué puedo asistirte?"),
    ("qué diferencia hay entre IDS e IPS?", "Un IDS detecta y alerta sobre actividad sospechosa, mientras que un IPS además puede bloquear el tráfico de forma activa."),
    ("gracias", "De nada. Si necesitas más ayuda con análisis de amenazas, herramientas o CVEs, dímelo."),
    ("qué es un zero-day?", "Una vulnerabilidad zero-day es una falla desconocida para el proveedor del software, sin parche disponible al momento de su explotación."),
    ("para qué sirve un SOC?", "Un SOC (Security Operations Center) centraliza la monitorización, detección y respuesta a incidentes de seguridad de una organización."),
    ("qué es phishing?", "El phishing es una técnica de ingeniería social donde un atacante suplanta a una entidad legítima para obtener credenciales o datos sensibles."),
]

# Adversariales: prompts cortos, ambiguos, jerga LATAM
ADVERSARIAL = [
    ("ese {cve}", "nvd_get_cve"),
    ("{cve} pls", "nvd_get_cve"),
    ("kev {cve}?", "cisa_kev_check"),
    ("vulns de {q}", "nvd_search"),
    ("checa {value}", "otx_check_ioc"),
    ("dale {desc}", "bash_exec"),
    ("rapido {cve}", "nvd_get_cve"),
    ("busqueme {q}", "nvd_search"),
]


def render_assistant(tool_name, args):
    return f'<|tool_call|>{{"name":"{tool_name}","args":{json.dumps(args, ensure_ascii=False)}}}<|/tool_call|>'


def gen_single_tool(rng, tool_name, n):
    out = []
    for _ in range(n):
        tpl = rng.choice(PHRASE_TEMPLATES[tool_name])
        if tool_name == "nvd_get_cve":
            cve = rng.choice(CVES)
            user = tpl.format(cve=cve)
            args = {"cve_id": cve}
        elif tool_name == "nvd_search":
            q = rng.choice(SEARCH_QUERIES)
            user = tpl.format(q=q)
            args = {"query": q, "limit": 10}
        elif tool_name == "cisa_kev_check":
            cve = rng.choice(CVES)
            user = tpl.format(cve=cve)
            args = {"cve_id": cve}
        elif tool_name == "otx_check_ioc":
            ioc_type = rng.choice(["ip", "domain", "hash"])
            pool = {"ip": IPS, "domain": DOMAINS, "hash": HASHES}[ioc_type]
            value = rng.choice(pool)
            user = tpl.format(value=value, ioc_type=ioc_type)
            args = {"ioc_type": ioc_type, "value": value}
        elif tool_name == "bash_exec":
            desc, cmd = rng.choice(BASH_TASKS)
            user = tpl.format(desc=desc)
            args = {"cmd": cmd}
        out.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": render_assistant(tool_name, args)},
            ]
        })
    return out


def gen_negatives(rng, n):
    out = []
    for _ in range(n):
        prompt, response = rng.choice(NEGATIVE_PROMPTS)
        out.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": prompt},
                {"role": "assistant", "content": response},
            ]
        })
    return out


def gen_adversarial(rng, n):
    out = []
    for _ in range(n):
        tpl, tool = rng.choice(ADVERSARIAL)
        if tool == "nvd_get_cve" or tool == "cisa_kev_check":
            cve = rng.choice(CVES)
            user = tpl.format(cve=cve)
            args = {"cve_id": cve}
        elif tool == "nvd_search":
            q = rng.choice(SEARCH_QUERIES)
            user = tpl.format(q=q)
            args = {"query": q, "limit": 10}
        elif tool == "otx_check_ioc":
            value = rng.choice(IPS + DOMAINS)
            ioc_type = "ip" if value in IPS else "domain"
            user = tpl.format(value=value)
            args = {"ioc_type": ioc_type, "value": value}
        elif tool == "bash_exec":
            desc, cmd = rng.choice(BASH_TASKS)
            user = tpl.format(desc=desc)
            args = {"cmd": cmd}
        out.append({
            "messages": [
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": user},
                {"role": "assistant", "content": render_assistant(tool, args)},
            ]
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1500, help="total examples")
    ap.add_argument("--out",  required=True, help="output JSONL path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--curriculum", action="store_true",
                    help="Fase 2: mezclar más adversariales y negativos hard")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Fase 1 (1.5k): 70% single-tool / 20% negativos / 10% adversariales
    # Fase 2 (8k):   55% single-tool / 20% negativos / 25% adversariales (curriculum)
    if args.curriculum:
        n_single = int(args.size * 0.55)
        n_neg    = int(args.size * 0.20)
        n_adv    = args.size - n_single - n_neg
    else:
        n_single = int(args.size * 0.70)
        n_neg    = int(args.size * 0.20)
        n_adv    = args.size - n_single - n_neg

    # Distribuir n_single equitativamente entre las 5 tools
    per_tool = n_single // 5
    examples = []
    for tool in PHRASE_TEMPLATES:
        examples.extend(gen_single_tool(rng, tool, per_tool))
    examples.extend(gen_negatives(rng, n_neg))
    examples.extend(gen_adversarial(rng, n_adv))

    rng.shuffle(examples)

    # Dedupe por hash de user prompt
    seen, deduped = set(), []
    for ex in examples:
        h = hashlib.md5(ex["messages"][1]["content"].encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(ex)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for ex in deduped:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[corpus] {len(deduped)} ejemplos → {out}")
    print(f"  single-tool: {n_single}  negativos: {n_neg}  adversariales: {n_adv}")


if __name__ == "__main__":
    main()
