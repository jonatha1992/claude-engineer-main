import difflib
import os
import re
import asyncio
import logging
import datetime
import json
import base64
from typing import Optional, Dict, Any, Tuple, List, Union
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
import groq
from groq import Groq
import sys
import signal
import venv
import io

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cargar variables de entorno
load_dotenv()

# Inicializar cliente Groq y consola Rich
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
console = Console()

# Variables globales
conversation_history: List[Dict[str, Any]] = []
file_contents: Dict[str, str] = {}
code_editor_memory: List[str] = []
code_editor_files: set = set()
automode: bool = False
running_processes: Dict[str, asyncio.subprocess.Process] = {}
MAX_CONTINUATION_ITERATIONS: int = 25
CONTINUATION_EXIT_PHRASE: str = "AUTOMODE_COMPLETE"
exit_automode: bool = False

# Modelos
MODEL: str = "llama-3.1-70b-versatile"

# Seguimiento de tokens
token_counters: Dict[str, Dict[str, int]] = {
    'main_model': {'input': 0, 'output': 0},
    'tool_checker': {'input': 0, 'output': 0},
    'code_editor': {'input': 0, 'output': 0},
    'code_execution': {'input': 0, 'output': 0}
}


# Prompts del sistema
BASE_SYSTEM_PROMPT_ES: str = '''
Eres Groq Engineer, un asistente de IA impulsado por modelos Groq, especializado en desarrollo de software con acceso a una variedad de herramientas y la capacidad de instruir y dirigir un agente de codificación y uno de ejecución de código. Tus capacidades incluyen:

1. Crear y gestionar estructuras de proyectos
2. Escribir, depurar y mejorar código en múltiples lenguajes
3. Proporcionar ideas arquitectónicas y aplicar patrones de diseño
4. Mantenerte actualizado con las últimas tecnologías y mejores prácticas
5. Analizar y manipular archivos dentro del directorio del proyecto
6. Ejecutar código y analizar su salida dentro de un entorno virtual aislado 'code_execution_env'
7. Gestionar y detener procesos en ejecución iniciados dentro del 'code_execution_env'

Herramientas disponibles y sus casos de uso óptimos:

1. create_folder: Crear nuevos directorios en la estructura del proyecto.
2. create_file: Generar nuevos archivos con contenido específico. Esfuérzate por hacer el archivo lo más completo y útil posible.
3. edit_and_apply: Examinar y modificar archivos existentes instruyendo a un agente de codificación separado.
4. execute_code: Ejecutar código Python exclusivamente en el entorno virtual 'code_execution_env' y analizar su salida.
5. stop_process: Detener un proceso en ejecución por su ID.
6. read_file: Leer el contenido de un archivo existente.
7. read_multiple_files: Leer el contenido de múltiples archivos existentes a la vez.
8. list_files: Listar todos los archivos y directorios en una carpeta específica.

Siempre busca la precisión, claridad y eficiencia en tus respuestas y acciones. Tus instrucciones deben ser precisas y completas. Al ejecutar código, recuerda siempre que se ejecuta en el entorno virtual aislado 'code_execution_env'. Ten en cuenta cualquier proceso de larga duración que inicies y gestiónalo adecuadamente, incluyendo detenerlos cuando ya no sean necesarios.
'''

AUTOMODE_SYSTEM_PROMPT_ES: str = '''
Actualmente estás en modo automático. Sigue estas pautas:

1. Establecimiento de Objetivos:
   - Establece objetivos claros y alcanzables basados en la solicitud del usuario.
   - Divide las tareas complejas en objetivos más pequeños y manejables.

2. Ejecución de Objetivos:
   - Trabaja en los objetivos sistemáticamente, utilizando las herramientas apropiadas para cada tarea.
   - Utiliza operaciones de archivos, escritura de código y ejecución según sea necesario.
   - Siempre lee un archivo antes de editarlo y revisa los cambios después de la edición.

3. Seguimiento del Progreso:
   - Proporciona actualizaciones regulares sobre el cumplimiento de los objetivos y el progreso general.
   - Utiliza la información de iteración para gestionar tu trabajo de manera efectiva.

4. Finalización del Modo Automático:
   - Cuando todos los objetivos se hayan completado, responde con "AUTOMODE_COMPLETE" para salir del modo automático.
   - No solicites tareas adicionales o modificaciones una vez que se hayan logrado los objetivos.

5. Conciencia de Iteración:
   - Tienes acceso a esta {{iteration_info}}.
   - Utiliza esta información para priorizar tareas y administrar el tiempo de manera efectiva.

Recuerda: Concéntrate en completar los objetivos establecidos de manera eficiente y efectiva. Evita conversaciones innecesarias o solicitudes de tareas adicionales.
'''

# Definición de herramientas
tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "create_folder",
            "description": "Crear una nueva carpeta en la ruta especificada",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "La ruta donde se debe crear la carpeta"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Crear un nuevo archivo en la ruta especificada con el contenido dado",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "La ruta donde se debe crear el archivo"},
                    "content": {"type": "string", "description": "El contenido del archivo"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_and_apply",
            "description": "Editar un archivo existente basado en instrucciones",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "La ruta del archivo a editar"},
                    "instructions": {"type": "string", "description": "Instrucciones para editar el archivo"},
                    "project_context": {"type": "string", "description": "Contexto sobre el proyecto para una mejor comprensión"}
                },
                "required": ["path", "instructions", "project_context"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Leer el contenido de un archivo",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "La ruta del archivo a leer"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Leer el contenido de múltiples archivos",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "items": {"type": "string"}, "description": "Las rutas de los archivos a leer"}
                },
                "required": ["paths"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Listar todos los archivos en un directorio",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "La ruta del directorio a listar (por defecto es el directorio actual)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Ejecutar código Python en el entorno aislado",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "El código Python a ejecutar"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_process",
            "description": "Detener un proceso en ejecución por su ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "process_id": {"type": "string", "description": "El ID del proceso a detener"}
                },
                "required": ["process_id"]
            }
        }
    }
]

async def get_user_input(prompt: str = "You: ") -> str:
    """Obtiene la entrada del usuario de manera asíncrona."""
    style = Style.from_dict({'prompt': 'cyan bold'})
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)

def setup_virtual_environment() -> Tuple[str, str]:
    """Configura el entorno virtual para la ejecución de código."""
    venv_name = "code_execution_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    try:
        if not os.path.exists(venv_path):
            venv.create(venv_path, with_pip=True)
        
        # Activar el entorno virtual
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")
        
        return venv_path, activate_script
    except Exception as e:
        logging.error(f"Error al configurar el entorno virtual: {str(e)}")
        raise

def update_system_prompt(current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> str:
    """Actualiza el prompt del sistema basado en el contexto actual."""
    global file_contents
    
    tools_prompt = """
    IMPORTANTE: Cuando necesites usar una herramienta, escribe el nombre de la herramienta seguido de los argumentos necesarios en una nueva línea. Por ejemplo:

    create_folder nombre_carpeta
    create_file nombre_archivo contenido_del_archivo
    read_file nombre_archivo
    list_files
    execute_code código_python_aquí
    edit_and_apply nombre_archivo instrucciones_de_edición
    read_multiple_files archivo1 archivo2 archivo3
    stop_process id_del_proceso

    No uses comillas ni formateo especial. Simplemente escribe el comando como se muestra arriba.
    Asegúrate de usar las herramientas cuando sea necesario para realizar acciones concretas.
    """
    
    file_contents_prompt = "\n\nContenido de los Archivos:\n"
    for path, content in file_contents.items():
        file_contents_prompt += f"\n--- {path} ---\n{content}\n"
    
    chain_of_thought_prompt = """
    Responde a la solicitud del usuario utilizando las herramientas relevantes (si están disponibles). Antes de llamar a una herramienta, realiza un análisis dentro de las etiquetas <thinking></thinking>. Primero, piensa en cuál de las herramientas proporcionadas es la relevante para responder a la solicitud del usuario. Segundo, revisa cada uno de los parámetros requeridos de la herramienta relevante y determina si el usuario ha proporcionado directamente o ha dado suficiente información para inferir un valor. Al decidir si el parámetro puede ser inferido, considera cuidadosamente todo el contexto para ver si respalda un valor específico. Si todos los parámetros requeridos están presentes o pueden ser razonablemente inferidos, cierra la etiqueta thinking y procede con la llamada a la herramienta. PERO, si falta uno de los valores para un parámetro requerido, NO invoques la función (ni siquiera con rellenos para los parámetros faltantes) y, en su lugar, pide al usuario que proporcione los parámetros faltantes. NO pidas más información sobre parámetros opcionales si no se proporciona.
    """
    
    if automode:
        iteration_info = ""
        if current_iteration is not None and max_iterations is not None:
            iteration_info = f"Actualmente estás en la iteración {current_iteration} de {max_iterations} en modo automático."
        automode_prompt = AUTOMODE_SYSTEM_PROMPT_ES.format(iteration_info=iteration_info)
        return BASE_SYSTEM_PROMPT_ES + tools_prompt + file_contents_prompt + "\n\n" + automode_prompt + "\n\n" + chain_of_thought_prompt
    else:
        return BASE_SYSTEM_PROMPT_ES + tools_prompt + file_contents_prompt + "\n\n" + chain_of_thought_prompt

def create_folder(path: str) -> str:
    """Crea una nueva carpeta en la ruta especificada."""
    try:
        os.makedirs(path, exist_ok=True)
        return f"Carpeta creada: {path}"
    except Exception as e:
        return f"Error al crear la carpeta: {str(e)}"

def create_file(path: str, content: str = "") -> str:
    """Crea un nuevo archivo con el contenido especificado."""
    global file_contents
    try:
        with open(path, 'w') as f:
            f.write(content)
        file_contents[path] = content
        return f"Archivo creado y añadido al prompt del sistema: {path}"
    except Exception as e:
        return f"Error al crear el archivo: {str(e)}"        
            


async def generate_edit_instructions(file_path: str, file_content: str, instructions: str, project_context: str, full_file_contents: Dict[str, str]) -> str:
    """Genera instrucciones de edición para un archivo."""
    global code_editor_tokens, code_editor_memory, code_editor_files
    
    # Asegurarse de que code_editor_tokens esté inicializado correctamente
    if 'code_editor_tokens' not in globals():
        global code_editor_tokens
        code_editor_tokens = {'input': 0, 'output': 0}
    
    try:
        memory_context = "\n".join([f"Memoria {i+1}:\n{mem}" for i, mem in enumerate(code_editor_memory)])
        full_file_contents_context = "\n\n".join([
            f"--- {path} ---\n{content}" for path, content in full_file_contents.items()
            if path != file_path or path not in code_editor_files
        ])

        system_prompt = f"""
        Eres un agente de IA de codificación que genera instrucciones de edición para archivos de código. Tu tarea es analizar el código proporcionado y generar bloques SEARCH/REPLACE para los cambios necesarios. Sigue estos pasos:

        1. Revisa todo el contenido del archivo para entender el contexto:
        {file_content}

        2. Analiza cuidadosamente las instrucciones específicas:
        {instructions}

        3. Ten en cuenta el contexto general del proyecto:
        {project_context}

        4. Considera la memoria de ediciones anteriores:
        {memory_context}

        5. Considera el contexto completo de todos los archivos en el proyecto:
        {full_file_contents_context}

        6. Genera bloques SEARCH/REPLACE para cada cambio necesario. Cada bloque debe:
           - Incluir suficiente contexto para identificar de manera única el código a cambiar
           - Proporcionar el código de reemplazo exacto, manteniendo la indentación y el formato correctos
           - Enfocarse en cambios específicos y dirigidos en lugar de modificaciones grandes y generales

        7. Asegúrate de que tus bloques SEARCH/REPLACE:
           - Aborden todos los aspectos relevantes de las instrucciones
           - Mantengan o mejoren la legibilidad y eficiencia del código
           - Consideren la estructura general y el propósito del código
           - Sigan las mejores prácticas y estándares de codificación para el lenguaje
           - Mantengan la consistencia con el contexto del proyecto y las ediciones anteriores
           - Tengan en cuenta el contexto completo de todos los archivos en el proyecto

        IMPORTANTE: DEVUELVE SOLO LOS BLOQUES SEARCH/REPLACE. SIN EXPLICACIONES NI COMENTARIOS.
        USA EL SIGUIENTE FORMATO PARA CADA BLOQUE:

        <SEARCH>
        Código a ser reemplazado
        </SEARCH>
        <REPLACE>
        Nuevo código a insertar
        </REPLACE>

        Si no se necesitan cambios, devuelve una lista vacía.
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Generate SEARCH/REPLACE blocks for the necessary changes."}
            ],
            model=MODEL,
            max_tokens=8000
        )

        if hasattr(response, 'usage'):
            code_editor_tokens['input'] += getattr(response.usage, 'prompt_tokens', 0)
            code_editor_tokens['output'] += getattr(response.usage, 'completion_tokens', 0)
        else:
            code_editor_tokens['input'] += len(system_prompt) + len("Generate SEARCH/REPLACE blocks for the necessary changes.")
            code_editor_tokens['output'] += len(response.choices[0].message.content) if response.choices else 0
        
        edit_instructions = parse_search_replace_blocks(response.choices[0].message.content)
        code_editor_memory.append(f"Edit Instructions for {file_path}:\n{response.choices[0].message.content}")
        code_editor_files.add(file_path)

        return edit_instructions

    except Exception as e:
        console.print(f"Error al generar instrucciones de edición: {str(e)}", style="bold red")
        return "[]"
def parse_search_replace_blocks(response_text: str) -> str:
    """Parsea los bloques SEARCH/REPLACE de la respuesta del modelo."""
    blocks = []
    pattern = r'<SEARCH>\n(.*?)\n</SEARCH>\n<REPLACE>\n(.*?)\n</REPLACE>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    for search, replace in matches:
        blocks.append({
            'search': search.strip(),
            'replace': replace.strip()
        })
    
    return json.dumps(blocks)

async def edit_and_apply(path: str, instructions: str, project_context: str, is_automode: bool = False, max_retries: int = 3) -> str:
    """Edita y aplica cambios a un archivo basado en instrucciones."""
    global file_contents
    try:
        original_content = file_contents.get(path, "")
        if not original_content:
            with open(path, 'r') as file:
                original_content = file.read()
            file_contents[path] = original_content

        for attempt in range(max_retries):
            edit_instructions_json = await generate_edit_instructions(path, original_content, instructions, project_context, file_contents)
            
            if edit_instructions_json:
                edit_instructions = json.loads(edit_instructions_json)
                console.print(Panel(f"Intento {attempt + 1}/{max_retries}: Se han generado los siguientes bloques SEARCH/REPLACE:", title="Instrucciones de Edición", style="cyan"))
                for i, block in enumerate(edit_instructions, 1):
                    console.print(f"Bloque {i}:")
                    console.print(Panel(f"SEARCH:\n{block['search']}\n\nREPLACE:\n{block['replace']}", expand=False))

                edited_content, changes_made, failed_edits = await apply_edits(path, edit_instructions, original_content)

                if changes_made:
                    file_contents[path] = edited_content
                    console.print(Panel(f"Contenido del archivo actualizado en el prompt del sistema: {path}", style="green"))
                    
                    if failed_edits:
                        console.print(Panel(f"Algunas ediciones no pudieron aplicarse. Reintentando...", style="yellow"))
                        instructions += f"\n\nPor favor, reintenta las siguientes ediciones que no pudieron aplicarse:\n{failed_edits}"
                        original_content = edited_content
                        continue
                    
                    return f"Cambios aplicados a {path}"
                elif attempt == max_retries - 1:
                    return f"No se pudieron aplicar cambios a {path} después de {max_retries} intentos. Por favor, revisa las instrucciones de edición e intenta de nuevo."
                else:
                    console.print(Panel(f"No se pudieron aplicar cambios en el intento {attempt + 1}. Reintentando...", style="yellow"))
            else:
                return f"No se sugirieron cambios para {path}"
        
        return f"Falló la aplicación de cambios a {path} después de {max_retries} intentos."
    except Exception as e:
        return f"Error al editar/aplicar al archivo: {str(e)}"

async def apply_edits(file_path: str, edit_instructions: List[Dict[str, str]], original_content: str) -> Tuple[str, bool, str]:
    """Aplica las ediciones al contenido del archivo."""
    changes_made = False
    edited_content = original_content
    total_edits = len(edit_instructions)
    failed_edits = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        edit_task = progress.add_task("[cyan]Aplicando ediciones...", total=total_edits)

        for i, edit in enumerate(edit_instructions, 1):
            search_content = edit['search'].strip()
            replace_content = edit['replace'].strip()
            
            pattern = re.compile(re.escape(search_content), re.DOTALL)
            match = pattern.search(edited_content)
            
            if match:
                start, end = match.span()
                replace_content_cleaned = re.sub(r'</?SEARCH>|</?REPLACE>', '', replace_content)
                edited_content = edited_content[:start] + replace_content_cleaned + edited_content[end:]
                changes_made = True
                
                diff_result = generate_diff(search_content, replace_content, file_path)
                console.print(Panel(diff_result, title=f"Cambios en {file_path} ({i}/{total_edits})", style="cyan"))
            else:
                console.print(Panel(f"Edición {i}/{total_edits} no aplicada: contenido no encontrado", style="yellow"))
                failed_edits.append(f"Edición {i}: {search_content}")

            progress.update(edit_task, advance=1)

        if not changes_made:
            console.print(Panel("No se aplicaron cambios. El contenido del archivo ya coincide con el estado deseado.", style="green"))
        else:
            with open(file_path, 'w') as file:
                file.write(edited_content)
            console.print(Panel(f"Los cambios se han escrito en {file_path}", style="green"))

        return edited_content, changes_made, "\n".join(failed_edits)

def generate_diff(original: str, new: str, path: str) -> Syntax:
    """Genera y resalta la diferencia entre dos cadenas de texto."""
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    diff_text = ''.join(diff)
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)

async def execute_code(code: str, timeout: int = 10) -> Tuple[str, str]:
    """Ejecuta código Python en un entorno virtual aislado."""
    global running_processes
    venv_path, activate_script = setup_virtual_environment()
    
    process_id = f"process_{len(running_processes)}"
    
    with open(f"{process_id}.py", "w") as f:
        f.write(code)
    
    if sys.platform == "win32":
        command = f'"{activate_script}" && python3 {process_id}.py'
    else:
        command = f'source "{activate_script}" && python3 {process_id}.py'
    
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        preexec_fn=None if sys.platform == "win32" else os.setsid
    )
    
    running_processes[process_id] = process
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode()
        stderr = stderr.decode()
        return_code = process.returncode
    except asyncio.TimeoutError:
        stdout = "Proceso iniciado y ejecutándose en segundo plano."
        stderr = ""
        return_code = "Ejecutando"
    
    execution_result = f"ID del Proceso: {process_id}\n\nSalida estándar:\n{stdout}\n\nSalida de error:\n{stderr}\n\nCódigo de retorno: {return_code}"
    return process_id, execution_result

def read_file(path: str) -> str:
    """Lee el contenido de un archivo."""
    global file_contents
    try:
        with open(path, 'r') as f:
            content = f.read()
        file_contents[path] = content
        return f"El archivo '{path}' ha sido leído y almacenado en el prompt del sistema."
    except Exception as e:
        return f"Error al leer el archivo: {str(e)}"

def read_multiple_files(paths: List[str]) -> str:
    """Lee el contenido de múltiples archivos."""
    global file_contents
    results = []
    for path in paths:
        try:
            with open(path, 'r') as f:
                content = f.read()
            file_contents[path] = content
            results.append(f"El archivo '{path}' ha sido leído y almacenado en el prompt del sistema.")
        except Exception as e:
            results.append(f"Error al leer el archivo '{path}': {str(e)}")
    return "\n".join(results)

def list_files(path: str = ".") -> str:
    """Lista los archivos en un directorio."""
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error al listar archivos: {str(e)}"

def stop_process(process_id: str) -> str:
    """Detiene un proceso en ejecución."""
    global running_processes
    if process_id in running_processes:
        process = running_processes[process_id]
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del running_processes[process_id]
        return f"El proceso {process_id} ha sido detenido."
    else:
        return f"No se encontró un proceso en ejecución con ID {process_id}."



async def send_to_ai_for_executing(code: str, execution_result: str) -> str:
    """Envía el código y su resultado de ejecución al modelo de AI para análisis."""
    global token_counters

    try:
        system_prompt = f"""
        Eres un agente de IA de ejecución de código. Tu tarea es analizar el código proporcionado y su resultado de ejecución del entorno virtual 'code_execution_env', luego proporcionar un resumen conciso de lo que funcionó, lo que no funcionó y cualquier observación importante. Sigue estos pasos:

        1. Revisa el código que se ejecutó en el entorno virtual 'code_execution_env':
        {code}

        2. Analiza el resultado de la ejecución del entorno virtual 'code_execution_env':
        {execution_result}

        3. Proporciona un breve resumen de:
           - Qué partes del código se ejecutaron con éxito en el entorno virtual
           - Cualquier error o comportamiento inesperado encontrado en el entorno virtual
           - Posibles mejoras o correcciones para los problemas, considerando la naturaleza aislada del entorno
           - Cualquier observación importante sobre el rendimiento o la salida del código dentro del entorno virtual
           - Si la ejecución se agotó por tiempo, explica lo que esto podría significar (por ejemplo, proceso de larga duración, bucle infinito)

        Sé conciso y céntrate en los aspectos más importantes de la ejecución del código dentro del entorno virtual 'code_execution_env'.

        IMPORTANTE: PROPORCIONA SOLO TU ANÁLISIS Y OBSERVACIONES. NO INCLUYAS DECLARACIONES DE PREFACIO NI EXPLICACIONES DE TU ROL.
        """

        response =  groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analiza esta ejecución de código del entorno virtual 'code_execution_env':\n\nCódigo:\n{code}\n\nResultado de Ejecución:\n{execution_result}"}
            ],
            model=MODEL,
            max_tokens=2000
        )

        if not response or not response.choices or not response.choices[0].message.content:
            raise ValueError("Se recibió una respuesta vacía o inválida de la API de Groq")

        if hasattr(response, 'usage'):
            token_counters['code_execution']['input'] += getattr(response.usage, 'prompt_tokens', 0)
            token_counters['code_execution']['output'] += getattr(response.usage, 'completion_tokens', 0)
        else:
            console.print("Los datos de uso de tokens no están disponibles en la respuesta.", style="yellow")

        analysis = response.choices[0].message.content if response.choices and response.choices[0].message and response.choices[0].message.content else "No hay análisis disponible."

        return analysis

    except Exception as e:
        console.print(f"Error en el análisis de ejecución de código de IA: {str(e)}", style="bold red")
        return f"Error al analizar la ejecución de código de 'code_execution_env': {str(e)}"






def extract_tool_calls(content: str) -> List[Dict[str, Any]]:
    tool_calls = []
    tool_pattern = r'`([a-zA-Z_]+)\s+([^`]+)`'
    matches = re.findall(tool_pattern, content)
    
    for i, (tool_name, args) in enumerate(matches):
        tool_calls.append({
            "id": f"call_{i}",
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": args.strip()
            }
        })
    
    return tool_calls


def reset_code_editor_memory():
    """Reinicia la memoria del editor de código."""
    global code_editor_memory
    code_editor_memory = []
    console.print(Panel("La memoria del editor de código ha sido reiniciada.", title="Reinicio", style="bold green"))

def reset_conversation():
    """Reinicia la conversación y todas las variables relacionadas."""
    global conversation_history, token_counters, file_contents, code_editor_files
    conversation_history = []
    token_counters = {
        'main_model': {'input': 0, 'output': 0},
        'tool_checker': {'input': 0, 'output': 0},
        'code_editor': {'input': 0, 'output': 0},
        'code_execution': {'input': 0, 'output': 0}
    }
    file_contents = {}
    code_editor_files = set()
    reset_code_editor_memory()
    console.print(Panel("El historial de conversación, contadores de tokens, contenidos de archivos, memoria del editor de código y archivos del editor de código han sido reiniciados.", title="Reinicio", style="bold green"))

def save_chat():
    """Guarda la conversación actual en un archivo Markdown."""
    now = datetime.datetime.now()
    filename = f"Chat_{now.strftime('%H%M')}.md"
    
    formatted_chat = "# Registro de Chat de Groq Engineer\n\n"
    for message in conversation_history:
        if message['role'] == 'user':
            formatted_chat += f"## Usuario\n\n{message['content']}\n\n"
        elif message['role'] == 'assistant':
            if isinstance(message['content'], str):
                formatted_chat += f"## Asistente\n\n{message['content']}\n\n"
            elif isinstance(message['content'], list):
                for content in message['content']:
                    if content['type'] == 'tool_use':
                        formatted_chat += f"### Uso de Herramienta: {content['name']}\n\n```json\n{json.dumps(content['input'], indent=2)}\n```\n\n"
                    elif content['type'] == 'text':
                        formatted_chat += f"## Asistente\n\n{content['text']}\n\n"
        elif message['role'] == 'tool':
            formatted_chat += f"### Resultado de Herramienta\n\n```\n{message['content']}\n```\n\n"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_chat)
    
    return filename


async def execute_tool(tool_call: Any) -> Dict[str, Any]:
    """Ejecuta una herramienta basada en la llamada de herramienta."""
    try:
        function_name = tool_call.function.name
        function_arguments = json.loads(tool_call.function.arguments)
        
        console.print(f"Ejecutando herramienta: {function_name}", style="bold cyan")
        console.print(f"Argumentos: {json.dumps(function_arguments, indent=2)}", style="cyan")
        
        result = None
        is_error = False

        if function_name == "create_folder":
            result = create_folder(function_arguments["path"])
        elif function_name == "create_file":
            result = create_file(function_arguments["path"], function_arguments.get("content", ""))
        elif function_name == "edit_and_apply":
            result = await edit_and_apply(
                function_arguments["path"],
                function_arguments["instructions"],
                function_arguments["project_context"],
                is_automode=automode
            )
        elif function_name == "read_file":
            result = read_file(function_arguments["path"])
        elif function_name == "read_multiple_files":
            result = read_multiple_files(function_arguments["paths"])
        elif function_name == "list_files":
            result = list_files(function_arguments.get("path", "."))
        elif function_name == "execute_code":
            process_id, execution_result = await execute_code(function_arguments["code"])
            analysis_task = asyncio.create_task(send_to_ai_for_executing(function_arguments["code"], execution_result))
            analysis = await analysis_task
            result = f"{execution_result}\n\nAnálisis:\n{analysis}"
            if process_id in running_processes:
                result += "\n\nNota: El proceso aún se está ejecutando en segundo plano."
        elif function_name == "stop_process":
            result = stop_process(function_arguments["process_id"])
        else:
            is_error = True
            result = f"Herramienta desconocida: {function_name}"

        console.print(f"Resultado de la herramienta: {result}", style="bold green")
        return {
            "content": result,
            "is_error": is_error
        }
    except Exception as e:
        error_message = f"Error al ejecutar la herramienta {function_name}: {str(e)}"
        logging.error(error_message)
        console.print(error_message, style="bold red")
        return {
            "content": error_message,
            "is_error": True
        }

async def chat_with_groq(user_input: str, current_iteration: Optional[int] = None, max_iterations: Optional[int] = None, max_retries: int = 3) -> Tuple[str, bool]:
    global conversation_history, token_counters

    current_conversation = [{"role": "user", "content": user_input}]
    messages = conversation_history + current_conversation

    assistant_response = ""

    for attempt in range(max_retries):
        try:
            system_message = {"role": "system", "content": update_system_prompt(current_iteration, max_iterations)}
            messages_with_system = [system_message] + messages

            console.print(Panel("Enviando solicitud a la API de Groq...", style="cyan"))
            chat_completion = groq_client.chat.completions.create(
                messages=messages_with_system,
                model=MODEL,
                max_tokens=8000,
                tools=tools
            )
            console.print(Panel("Respuesta recibida de la API de Groq", style="green"))
            

            if not chat_completion or not chat_completion.choices:
                raise ValueError("Se recibió una respuesta vacía de la API de Groq")

            assistant_message = chat_completion.choices[0].message
            
            if assistant_message.content:
                assistant_response = assistant_message.content
            elif assistant_message.tool_calls:
                assistant_response = "Se detectaron llamadas a herramientas:"
                for tool_call in assistant_message.tool_calls:
                    assistant_response += f"\n- {tool_call.function.name}: {tool_call.function.arguments}"
            else:
                console.print(f"Intento {attempt + 1}/{max_retries} fallido: respuesta vacía. Reintentando...", style="bold yellow")
                continue

            # Mostrar la respuesta del asistente
            console.print(Panel(Markdown(assistant_response), title="Respuesta de Groq", title_align="left", border_style="blue", expand=False))
            
            # Procesar tool_calls
            if assistant_message.tool_calls:
                console.print(Panel("Llamadas a herramientas detectadas", title="Uso de Herramientas", style="bold yellow"))
                console.print(Panel(json.dumps([tc.model_dump() for tc in assistant_message.tool_calls], indent=2), title="Llamadas a Herramientas", style="cyan"))

                for tool_call in assistant_message.tool_calls:
                    tool_result = await execute_tool(tool_call)

                    if tool_result["is_error"]:
                        console.print(Panel(tool_result["content"], title="Error en la Ejecución de la Herramienta", style="bold red"))
                    else:
                        console.print(Panel(tool_result["content"], title_align="left", title="Resultado de la Herramienta", style="green"))

                    current_conversation.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tool_call.model_dump()]
                    })

                    current_conversation.append({
                        "role": "tool",
                        "content": tool_result["content"],
                        "tool_call_id": tool_call.id
                    })

            messages = conversation_history + current_conversation

            try:
                tool_response = groq_client.chat.completions.create(
                    messages=messages,
                    model=MODEL,
                    max_tokens=8000,
                    tools=tools
                )
                if hasattr(tool_response, 'usage'):
                    token_counters['tool_checker']['input'] += getattr(tool_response.usage, 'prompt_tokens', 0)
                    token_counters['tool_checker']['output'] += getattr(tool_response.usage, 'completion_tokens', 0)

                tool_checker_response = tool_response.choices[0].message.content if tool_response.choices else None
                if tool_checker_response:
                    console.print(Panel(Markdown(tool_checker_response), title="Respuesta de Groq al Resultado de la Herramienta", title_align="left", border_style="blue", expand=False))
                    assistant_response += "\n\n" + tool_checker_response
            except Exception as e:
                error_message = f"Error en la respuesta de la herramienta: {str(e)}"
                console.print(Panel(error_message, title="Error", style="bold red"))
                assistant_response += f"\n\n{error_message}"

            break  # Exit loop if successful

        except Exception as e:
            console.print(Panel(f"Error de API: {str(e)}", title="Error de API", style="bold red"))
            if attempt == max_retries - 1:
                return "Lo siento, hubo un error al comunicarse con la IA. Por favor, intenta de nuevo.", False

    conversation_history = messages + [{"role": "assistant", "content": assistant_response}]

    return assistant_response, CONTINUATION_EXIT_PHRASE in assistant_response

async def main():
    """Función principal que maneja la interacción con el usuario."""
    global automode, conversation_history, exit_automode
    console.print(Panel("¡Bienvenido al Chat de Groq Engineer!", title="Bienvenida", style="bold green"))
    console.print("Escribe 'salir' para terminar la conversación.")
    console.print("Escribe 'automode [número]' para entrar en modo Autónomo con un número específico de iteraciones.")
    console.print("Escribe 'reset' para borrar el historial de la conversación.")
    console.print("Escribe 'guardar chat' para guardar la conversación en un archivo Markdown.")
    console.print("Mientras estés en automode, presiona Ctrl+C en cualquier momento para salir del bucle de automode y volver al chat regular.")

    signal.signal(signal.SIGINT, lambda signum, frame: setattr(sys.modules[__name__], 'exit_automode', True))

    while True:
        try:
            user_input = await get_user_input()

            if user_input.lower() == 'salir':
                console.print(Panel("Gracias por chatear. ¡Hasta luego!", title="Despedida", style="bold green"))
                break

            if user_input.lower() == 'reset':
                reset_conversation()
                continue

            if user_input.lower() == 'guardar chat':
                filename = save_chat()
                console.print(Panel(f"Chat guardado en {filename}", title="Chat Guardado", style="bold green"))
                continue

            if user_input.lower().startswith('automode'):
                try:
                    parts = user_input.split()
                    if len(parts) > 1 and parts[1].isdigit():
                        max_iterations = int(parts[1])
                    else:
                        max_iterations = MAX_CONTINUATION_ITERATIONS

                    automode = True
                    exit_automode = False
                    console.print(Panel(f"Entrando en modo automático con {max_iterations} iteraciones. Por favor, proporciona el objetivo del modo automático.", title="Modo Automático", style="bold yellow"))
                    console.print(Panel("Presiona Ctrl+C en cualquier momento para salir del bucle de modo automático.", style="bold yellow"))
                    user_input = await get_user_input()

                    iteration_count = 0
                    while automode and iteration_count < max_iterations and not exit_automode:
                        console.print(Panel(f"Iniciando iteración {iteration_count + 1}/{max_iterations}", title="Modo Automático", style="bold cyan"))
                        
                        try:
                            response, exit_continuation = await chat_with_groq(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)
                            
                            console.print(Panel(f"Iteración {iteration_count + 1} completada", title="Modo Automático", style="bold green"))
                            console.print(Panel(Markdown(response), title=f"Respuesta de la Iteración {iteration_count + 1}", style="cyan"))

                            if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                                console.print(Panel("Modo automático completado.", title="Modo Automático", style="green"))
                                automode = False
                            else:
                                console.print(Panel(f"Continuando con la siguiente iteración. Presiona Ctrl+C para salir del modo automático.", title="Modo Automático", style="yellow"))
                                user_input = "Continúa con el siguiente paso. O DETENTE diciendo 'AUTOMODE_COMPLETE' si crees que has logrado los resultados establecidos en la solicitud original."
                            
                            iteration_count += 1

                            if iteration_count >= max_iterations:
                                console.print(Panel("Máximo de iteraciones alcanzado. Saliendo del modo automático.", title="Modo Automático", style="bold red"))
                                automode = False

                        except Exception as e:
                            error_message = f"Error en la iteración {iteration_count + 1}: {str(e)}"
                            console.print(Panel(error_message, title="Error en Modo Automático", style="bold red"))
                            logging.error(error_message)
                            
                            # Preguntar al usuario si desea continuar o salir del modo automático
                            user_choice = await get_user_input("¿Deseas continuar con el modo automático? (s/n): ")
                            if user_choice.lower() != 's':
                                console.print(Panel("Saliendo del modo automático debido al error.", title="Modo Automático", style="bold red"))
                                automode = False
                                break

                    if exit_automode:
                        console.print(Panel("\nModo automático interrumpido por el usuario. Saliendo del modo automático.", title="Modo Automático", style="bold red"))
                        automode = False
                        if conversation_history and conversation_history[-1]["role"] == "user":
                            conversation_history.append({"role": "assistant", "content": "Modo automático interrumpido. ¿Cómo puedo ayudarte más?"})

                except Exception as e:
                    console.print(Panel(f"\nError en modo automático: {str(e)}", title="Error de Modo Automático", style="bold red"))
                    logging.error(f"Error en modo automático: {str(e)}")
                    automode = False

                console.print(Panel("Salido del modo automático. Volviendo al chat regular.", style="green"))
            else:
                response, _ = await chat_with_groq(user_input)
                console.print(Panel(Markdown(response), title="Respuesta de Groq", style="cyan"))

        except Exception as e:
            error_message = f"Error general en la ejecución: {str(e)}"
            console.print(Panel(error_message, title="Error", style="bold red"))
            logging.error(error_message)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\nPrograma terminado por el usuario.", style="bold yellow")
    except Exception as e:
        console.print(f"\nError inesperado: {str(e)}", style="bold red")
        logging.error(f"Error inesperado: {str(e)}")


