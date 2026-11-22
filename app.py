import gc
import json
import os
import shutil
import time
import uuid
from datetime import datetime

import streamlit as st
import torch
import transformers
from fpdf import FPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from config import APP_CONFIG

transformers.logging.set_verbosity_error()

# Limita hilos de CPU para contener el uso de memoria en contenedores con RAM restringida
os.environ.setdefault("OMP_NUM_THREADS", str(APP_CONFIG.model.cpu_threads))
os.environ.setdefault("MKL_NUM_THREADS", str(APP_CONFIG.model.cpu_threads))
try:
    torch.set_num_threads(APP_CONFIG.model.cpu_threads)
except RuntimeError:
    pass


# ==============================================================================
# 1. MEMORIA A CORTO PLAZO, PERFILADO ANÓNIMO Y AUDITORÍA SIN PII
# ==============================================================================
class SimpleChatMemory:
    def __init__(self, max_messages: int = 6):
        self.history = []
        self.max_messages = max_messages

    def add_user_message(self, message: str):
        self.history.append(f"Usuario: {message}")

    def add_ai_message(self, message: str):
        self.history.append(f"Asistente: {message}")

    def get_history_string(self) -> str:
        return "\n".join(self.history[-self.max_messages :])

    def clear(self):
        self.history = []


class UserProfileManager:
    def __init__(self, filepath: str, default_os: str, default_format: str):
        self.filepath = filepath
        self.default_os = default_os
        self.default_format = default_format
        self.profile = self._load_or_create()

    def _load_or_create(self) -> dict:
        default_profile = {
            "session_id": str(uuid.uuid4())[:8],
            "sistema_operativo": self.default_os,
            "formato_preferido": self.default_format,
        }
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    return data
            except (json.JSONDecodeError, OSError):
                pass

        self.save(default_profile)
        return default_profile

    def save(self, data: dict | None = None):
        if data is not None:
            self.profile = data
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.profile, f, indent=4, ensure_ascii=False)

    def update_os(self, new_os: str):
        self.profile["sistema_operativo"] = new_os
        self.save()

    def get_profile_context(self) -> str:
        return (
            f"SO del usuario: {self.profile['sistema_operativo']} | "
            f"Preferencia: {self.profile['formato_preferido']}"
        )


class SupportLogger:
    def __init__(self, log_filepath: str, min_valid_len: int = 15):
        self.log_filepath = log_filepath
        self.min_valid_len = min_valid_len

    def log_interaction(
        self,
        session_id: str,
        question: str,
        response: str,
        latency_sec: float,
        docs_count: int,
    ):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "latencia_segundos": round(latency_sec, 2),
            "fragmentos_recuperados": docs_count,
            "consulta": question,
            "respuesta_generada": response,
            "estado": "EXITO" if len(response.strip()) > self.min_valid_len else "POSIBLE_FALLA",
        }
        with open(self.log_filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


# ==============================================================================
# 2. INICIALIZACIÓN CACHEADA DEL RAG Y DEL MODELO LOCAL
# ==============================================================================
def ensure_manual_pdf(pdf_path: str):
    if os.path.exists(pdf_path):
        return

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    contenido_manual = """MANUAL OFICIAL DE SOPORTE TECNICO TI - NIVEL 1

1. CONEXION A RED WI-FI CORPORATIVA
Para conectar su equipo portatil a la red Wi-Fi corporativa 'Empresa_Secure', vaya a Configuracion > Red e Internet > Wi-Fi, seleccione la red e introduzca su contrasena corporativa de Active Directory. Si el equipo no detecta la red, reinicie el adaptador de red en el panel de control.

2. CONFIGURACION DE CORREO OFFICE 365 EN MOVILES
Para configurar el correo corporativo de Office 365 en dispositivos moviles Android o iOS, descargue la aplicacion oficial Microsoft Outlook desde Google Play Store o App Store. Inicie sesion con su correo completo y contrasena temporal.

3. IMPRESORAS DE RED COMPARTIDAS EN WINDOWS
Si experimenta problemas con impresoras de red compartidas en Windows, vaya a Configuracion > Bluetooth y dispositivos > Impresoras y escaneres, haga clic en 'Agregar dispositivo' y seleccione la impresora de red con el prefijo 'PIS_'. Si persiste el error, reinicie el servicio 'Cola de impresion'.

4. SINCRONIZACION DE ONEDRIVE EN MACOS
Para sincronizar carpetas en la nube con OneDrive en macOS, abra Finder, ubique la carpeta de OneDrive e inicie sesion con sus credenciales de Office 365 para activar la sincronizacion automatica de archivos corporativos.

5. RESTABLECIMIENTO DE CONTRASENA DE VPN CORPORATIVA
Para restablecer su contrasena de acceso remoto VPN (GlobalProtect), ingrese al portal de autogestion https://clave.empresa.local desde un navegador web, seleccione 'Olvide mi contrasena' y valide su identidad mediante el codigo enviado a su aplicacion Microsoft Authenticator.
"""
    pdf.multi_cell(0, 8, contenido_manual)
    pdf.output(pdf_path)


def _resolve_torch_dtype(dtype_name: str):
    normalized = (dtype_name or "float16").lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    return torch.float16


@st.cache_resource(show_spinner="Cargando base de conocimiento corporativa...")
def build_retriever():
    ensure_manual_pdf(APP_CONFIG.rag.pdf_path)

    loader = PyPDFLoader(APP_CONFIG.rag.pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=APP_CONFIG.rag.chunk_size,
        chunk_overlap=APP_CONFIG.rag.chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=APP_CONFIG.model.embedding_model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 8},
    )

    persist_dir = APP_CONFIG.rag.persist_directory
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=APP_CONFIG.rag.collection_name,
        persist_directory=persist_dir,
    )

    del docs, chunks
    gc.collect()

    return vectorstore.as_retriever(search_kwargs={"k": APP_CONFIG.rag.top_k})


@st.cache_resource(show_spinner="Cargando modelo local de soporte TI...")
def build_llm():
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else _resolve_torch_dtype(APP_CONFIG.model.torch_dtype)

    tokenizer = AutoTokenizer.from_pretrained(APP_CONFIG.model.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        APP_CONFIG.model.model_id,
        dtype=dtype,
        low_cpu_mem_usage=APP_CONFIG.model.low_cpu_mem_usage,
        device_map="auto" if use_cuda else None,
    )
    model.eval()

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=APP_CONFIG.model.max_new_tokens,
        temperature=APP_CONFIG.model.temperature,
        do_sample=APP_CONFIG.model.do_sample,
        repetition_penalty=APP_CONFIG.model.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )

    gc.collect()
    return HuggingFacePipeline(pipeline=gen_pipe)


SYSTEM_TEMPLATE = """<|system|>
Eres un Asistente de Soporte TI corporativo de Nivel 1.
REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE EN ESPAÑOL.
2. Extrae SOLO los pasos literales del "Contexto oficial" (máximo 2 o 3 pasos cortos). Detente inmediatamente al terminar la información del texto.

Perfil del usuario: {user_profile}

Contexto oficial:
{context}

Historial:
{chat_history}
<|user|>
{question}
<|assistant|>
Según el manual oficial de soporte TI:
"""

PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["user_profile", "context", "chat_history", "question"],
    template=SYSTEM_TEMPLATE,
)


def generate_support_response(
    question: str,
    retriever,
    llm,
    memory: SimpleChatMemory,
    profile_mgr: UserProfileManager,
    logger: SupportLogger,
) -> str:
    start_ts = time.time()

    docs = retriever.invoke(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    history_text = memory.get_history_string()
    profile_text = profile_mgr.get_profile_context()

    prompt_final = PROMPT_TEMPLATE.format(
        user_profile=profile_text,
        context=context_text,
        chat_history=history_text,
        question=question,
    )

    raw_output = llm.invoke(prompt_final).strip()

    if "<|assistant|>" in raw_output:
        answer = raw_output.split("<|assistant|>")[-1].strip()
    else:
        answer = raw_output

    if "<|user|>" in answer:
        answer = answer.split("<|user|>")[0].strip()

    if answer and answer[-1] not in ".!?":
        last_dot = answer.rfind(".")
        if last_dot != -1:
            answer = answer[: last_dot + 1]

    memory.add_user_message(question)
    memory.add_ai_message(answer)

    elapsed = time.time() - start_ts
    logger.log_interaction(
        session_id=profile_mgr.profile["session_id"],
        question=question,
        response=answer,
        latency_sec=elapsed,
        docs_count=len(docs),
    )
    return answer


# ==============================================================================
# 3. INTERFAZ STREAMLIT Y ORIENTACIÓN AL USUARIO
# ==============================================================================
st.set_page_config(
    page_title=APP_CONFIG.app_title,
    page_icon="🛠️",
    layout="wide",
)

if "memory" not in st.session_state:
    st.session_state.memory = SimpleChatMemory(max_messages=APP_CONFIG.model.max_history_messages)
if "profile_mgr" not in st.session_state:
    st.session_state.profile_mgr = UserProfileManager(
        filepath=APP_CONFIG.governance.profile_filepath,
        default_os=APP_CONFIG.governance.default_os,
        default_format=APP_CONFIG.governance.default_format,
    )
if "logger" not in st.session_state:
    st.session_state.logger = SupportLogger(
        log_filepath=APP_CONFIG.governance.log_filepath,
        min_valid_len=APP_CONFIG.governance.min_valid_response_len,
    )
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Guía del Asistente TI")
    st.markdown(
        "**Capacidades:**\n"
        "- Guía paso a paso para procedimientos de Nivel 1 documentados en el manual corporativo (Wi-Fi, Outlook móvil, impresoras de red, OneDrive en macOS y restablecimiento de VPN).\n"
        "- Conversación con memoria de turno reciente y registro anónimo sin datos personales (PII)."
    )
    st.markdown(
        "**Limitaciones:**\n"
        "- Solo responde con base en la documentación cargada; no reemplaza a un técnico especialista para fallas de hardware o permisos complejos.\n"
        "- Al usar un modelo compacto local, puede omitir matices si la pregunta es muy ambigua."
    )

    st.divider()
    st.subheader("Indicaciones recomendadas")
    sample_prompts = [
        "¿Cómo me conecto a la red Wi-Fi corporativa?",
        "¿Cómo configuro el correo de Office 365 en mi celular?",
        "¿Cómo sincronizo OneDrive en macOS?",
        "¿Cómo restablezco la contraseña de la VPN?",
    ]
    selected_sample = None
    for idx, sample in enumerate(sample_prompts):
        if st.button(sample, key=f"sample_{idx}", use_container_width=True):
            selected_sample = sample

    st.divider()
    current_os = st.session_state.profile_mgr.profile.get(
        "sistema_operativo", APP_CONFIG.governance.default_os
    )
    os_choice = st.selectbox(
        "Sistema operativo de referencia",
        options=["Windows 11", "Windows 10", "macOS"],
        index=["Windows 11", "Windows 10", "macOS"].index(current_os)
        if current_os in ["Windows 11", "Windows 10", "macOS"]
        else 0,
    )
    if os_choice != current_os:
        st.session_state.profile_mgr.update_os(os_choice)

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

    st.divider()
    st.warning(
        "**Aviso legal:** Este prototipo es una herramienta de apoyo basada en documentación interna y un modelo de lenguaje compacto. "
        "Puede presentar imprecisiones, omisiones o errores de interpretación. Verifique siempre los pasos críticos y contacte a Soporte TI si persiste el incidente."
    )

st.title(APP_CONFIG.app_title)
st.caption(APP_CONFIG.app_subtitle)

retriever = build_retriever()
llm = build_llm()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt_input = st.chat_input(
    "Escribe tu consulta de soporte TI (ej. ¿Cómo configuro Outlook en el celular?)"
)
user_query = selected_sample or prompt_input

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Consultando el manual oficial de soporte TI..."):
            reply = generate_support_response(
                question=user_query,
                retriever=retriever,
                llm=llm,
                memory=st.session_state.memory,
                profile_mgr=st.session_state.profile_mgr,
                logger=st.session_state.logger,
            )
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})