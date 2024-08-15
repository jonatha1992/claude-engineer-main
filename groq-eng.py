import os
import re
import asyncio
import difflib
import logging
import datetime
import json
import base64
from io import BytesIO
from PIL import Image
from typing import Optional, Dict, Any, Tuple, List
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.box import ROUNDED
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from groq import Groq
import subprocess
import sys
import signal
import venv
import io

# Load environment variables
load_dotenv()

# Initialize Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
console = Console()

# Global state
conversation_history = []
file_contents = {}
code_editor_memory = []
code_editor_files = set()
automode = False
running_processes = {}
MAX_CONTINUATION_ITERATIONS = 25
MAX_CONTEXT_TOKENS = 200000
CONTINUATION_EXIT_PHRASE = "AUTOMODE_COMPLETE"
exit_automode = False  # New global variable to control automode exit

# Models
MODEL = "llama-3.1-8b-instant"  # Reemplaza con un modelo válido y accesible

MAINMODEL = MODEL
TOOLCHECKERMODEL = MODEL
CODEEDITORMODEL = MODEL
CODEEXECUTIONMODEL = MODEL

# Token tracking
main_model_tokens = {'input': 0, 'output': 0}
tool_checker_tokens = {'input': 0, 'output': 0}
code_editor_tokens = {'input': 0, 'output': 0}
code_execution_tokens = {'input': 0, 'output': 0}

# System prompts


BASE_SYSTEM_PROMPT_ES = '''
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
AUTOMODE_SYSTEM_PROMPT_ES = '''
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
tools = [
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

async def get_user_input(prompt="You: "):
    style = Style.from_dict({'prompt': 'cyan bold'})
    session = PromptSession(style=style)
    return await session.prompt_async(prompt, multiline=False)

def setup_virtual_environment() -> Tuple[str, str]:
    venv_name = "code_execution_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    try:
        if not os.path.exists(venv_path):
            venv.create(venv_path, with_pip=True)
        
        # Activate the virtual environment
        if sys.platform == "win32":
            activate_script = os.path.join(venv_path, "Scripts", "activate.bat")
        else:
            activate_script = os.path.join(venv_path, "bin", "activate")
        
        return venv_path, activate_script
    except Exception as e:
        logging.error(f"Error setting up virtual environment: {str(e)}")
        raise

def update_system_prompt(current_iteration: Optional[int] = None, max_iterations: Optional[int] = None) -> str:
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

def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"

def create_file(path, content=""):
    global file_contents
    try:
        with open(path, 'w') as f:
            f.write(content)
        file_contents[path] = content
        return f"File created and added to system prompt: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"

def highlight_diff(diff_text):
    return Syntax(diff_text, "diff", theme="monokai", line_numbers=True)

async def generate_edit_instructions(file_path, file_content, instructions, project_context, full_file_contents):
    global code_editor_tokens, code_editor_memory, code_editor_files
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
            model=CODEEDITORMODEL,
            max_tokens=8000
        )

    
        if hasattr(response, 'usage'):
            code_editor_tokens['input'] += getattr(response.usage, 'prompt_tokens', 0)
            code_editor_tokens['output'] += getattr(response.usage, 'completion_tokens', 0)
        else:
            # Si no hay información de uso, hacemos una estimación basada en la longitud
            code_editor_tokens['input'] += len(system_prompt) + len("Generate SEARCH/REPLACE blocks for the necessary changes.")
            code_editor_tokens['output'] += len(response.choices[0].message.content) if response.choices else 0
        edit_instructions = parse_search_replace_blocks(response.choices[0].message.content)
        code_editor_memory.append(f"Edit Instructions for {file_path}:\n{response.choices[0].message.content}")
        code_editor_files.add(file_path)

        return edit_instructions

    except Exception as e:
        console.print(f"Error in generating edit instructions: {str(e)}", style="bold red")
        return []

def parse_search_replace_blocks(response_text):
    blocks = []
    pattern = r'<SEARCH>\n(.*?)\n</SEARCH>\n<REPLACE>\n(.*?)\n</REPLACE>'
    matches = re.findall(pattern, response_text, re.DOTALL)
    
    for search, replace in matches:
        blocks.append({
            'search': search.strip(),
            'replace': replace.strip()
        })
    
    return json.dumps(blocks)

async def edit_and_apply(path, instructions, project_context, is_automode=False, max_retries=3):
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
                console.print(Panel(f"Attempt {attempt + 1}/{max_retries}: The following SEARCH/REPLACE blocks have been generated:", title="Edit Instructions", style="cyan"))
                for i, block in enumerate(edit_instructions, 1):
                    console.print(f"Block {i}:")
                    console.print(Panel(f"SEARCH:\n{block['search']}\n\nREPLACE:\n{block['replace']}", expand=False))

                edited_content, changes_made, failed_edits = await apply_edits(path, edit_instructions, original_content)

                if changes_made:
                    file_contents[path] = edited_content
                    console.print(Panel(f"File contents updated in system prompt: {path}", style="green"))
                    
                    if failed_edits:
                        console.print(Panel(f"Some edits could not be applied. Retrying...", style="yellow"))
                        instructions += f"\n\nPlease retry the following edits that could not be applied:\n{failed_edits}"
                        original_content = edited_content
                        continue
                    
                    return f"Changes applied to {path}"
                elif attempt == max_retries - 1:
                    return f"No changes could be applied to {path} after {max_retries} attempts. Please review the edit instructions and try again."
                else:
                    console.print(Panel(f"No changes could be applied in attempt {attempt + 1}. Retrying...", style="yellow"))
            else:
                return f"No changes suggested for {path}"
        
        return f"Failed to apply changes to {path} after {max_retries} attempts."
    except Exception as e:
        return f"Error editing/applying to file: {str(e)}"
    
async def apply_edits(file_path, edit_instructions, original_content):
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
        edit_task = progress.add_task("[cyan]Applying edits...", total=total_edits)

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
                console.print(Panel(diff_result, title=f"Changes in {file_path} ({i}/{total_edits})", style="cyan"))
            else:
                console.print(Panel(f"Edit {i}/{total_edits} not applied: content not found", style="yellow"))
                failed_edits.append(f"Edit {i}: {search_content}")

            progress.update(edit_task, completed=1)
            progress.update(edit_task, advance=1)

        if not changes_made:
            console.print(Panel("No changes were applied. The file content already matches the desired state.", style="green"))
        else:
            with open(file_path, 'w') as file:
                file.write(edited_content)
            console.print(Panel(f"Changes have been written to {file_path}", style="green"))

        return edited_content, changes_made, "\n".join(failed_edits)

def generate_diff(original, new, path):
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3
    ))

    diff_text = ''.join(diff)
    highlighted_diff = highlight_diff(diff_text)

    return highlighted_diff

async def execute_code(code, timeout=10):
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
        stdout = "Process started and running in the background."
        stderr = ""
        return_code = "Running"
    
    execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
    return process_id, execution_result



def read_file(path):
    global file_contents
    try:
        with open(path, 'r') as f:
            content = f.read()
        file_contents[path] = content
        return f"File '{path}' has been read and stored in the system prompt."
    except Exception as e:
        return f"Error reading file: {str(e)}"

def read_multiple_files(paths):
    global file_contents
    results = []
    for path in paths:
        try:
            with open(path, 'r') as f:
                content = f.read()
            file_contents[path] = content
            results.append(f"File '{path}' has been read and stored in the system prompt.")
        except Exception as e:
            results.append(f"Error reading file '{path}': {str(e)}")
    return "\n".join(results)

def list_files(path="."):
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def stop_process(process_id):
    global running_processes
    if process_id in running_processes:
        process = running_processes[process_id]
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del running_processes[process_id]
        return f"Process {process_id} has been stopped."
    else:
        return f"No running process found with ID {process_id}."


async def execute_tool(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    try:
        function = tool_call.get('function', {})
        function_name = function.get('name')
        function_arguments = json.loads(function.get('arguments', '{}'))
        
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
            result = f"{execution_result}\n\nAnalysis:\n{analysis}"
            if process_id in running_processes:
                result += "\n\nNote: The process is still running in the background."
        elif function_name == "stop_process":
            result = stop_process(function_arguments["process_id"])
        else:
            is_error = True
            result = f"Unknown tool: {function_name}"

        return {
            "content": result,
            "is_error": is_error
        }
    except KeyError as e:
        logging.error(f"Missing required parameter {str(e)} for tool {function_name}")
        return {
            "content": f"Error: Missing required parameter {str(e)} for tool {function_name}",
            "is_error": True
        }
    except Exception as e:
        logging.error(f"Error executing tool {function_name}: {str(e)}")
        return {
            "content": f"Error executing tool {function_name}: {str(e)}",
            "is_error": True
        }

async def send_to_ai_for_executing(code, execution_result):
    global code_execution_tokens

    try:
        system_prompt = f"""
        You are an AI code execution agent. Your task is to analyze the provided code and its execution result from the 'code_execution_env' virtual environment, then provide a concise summary of what worked, what didn't work, and any important observations. Follow these steps:

        1. Review the code that was executed in the 'code_execution_env' virtual environment:
        {code}

        2. Analyze the execution result from the 'code_execution_env' virtual environment:
        {execution_result}

        3. Provide a brief summary of:
           - What parts of the code executed successfully in the virtual environment
           - Any errors or unexpected behavior encountered in the virtual environment
           - Potential improvements or fixes for issues, considering the isolated nature of the environment
           - Any important observations about the code's performance or output within the virtual environment
           - If the execution timed out, explain what this might mean (e.g., long-running process, infinite loop)

        Be concise and focus on the most important aspects of the code execution within the 'code_execution_env' virtual environment.

        IMPORTANT: PROVIDE ONLY YOUR ANALYSIS AND OBSERVATIONS. DO NOT INCLUDE ANY PREFACING STATEMENTS OR EXPLANATIONS OF YOUR ROLE.
        """

        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this code execution from the 'code_execution_env' virtual environment:\n\nCode:\n{code}\n\nExecution Result:\n{execution_result}"}
            ],
            model=CODEEXECUTIONMODEL,
            max_tokens=2000
        )

        # Ensure response is valid before accessing usage and content
        if not response or not response.choices or not response.choices[0].message.content:
            raise ValueError("Received empty or invalid response from Groq API")

        if hasattr(response, 'usage'):
            code_execution_tokens['input'] += getattr(response.usage, 'prompt_tokens', 0)
            code_execution_tokens['output'] += getattr(response.usage, 'completion_tokens', 0)
        else:
            console.print("Token usage data is not available in the response.", style="yellow")

        analysis = response.choices[0].message.content if response.choices and response.choices[0].message and response.choices[0].message.content else "No analysis available."

        return analysis

    except Exception as e:
        console.print(f"Error in AI code execution analysis: {str(e)}", style="bold red")
        return f"Error analyzing code execution from 'code_execution_env': {str(e)}"

# async def chat_with_groq(user_input, image_path=None, current_iteration=None, max_iterations=None):
#     global conversation_history, main_model_tokens

#     current_conversation = []

#     if image_path:
#         image_base64 = encode_image_to_base64(image_path)
#         if image_base64.startswith("Error"):
#             console.print(Panel(f"Error encoding image: {image_base64}", title="Error", style="bold red"))
#             return "I'm sorry, there was an error processing the image. Please try again.", False

#         image_message = {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "image",
#                     "source": {
#                         "type": "base64",
#                         "media_type": "image/jpeg",
#                         "data": image_base64
#                     }
#                 },
#                 {
#                     "type": "text",
#                     "text": f"User input for image: {user_input}"
#                 }
#             ]
#         }
#         current_conversation.append(image_message)
#     else:
#         current_conversation.append({"role": "user", "content": user_input})

#     messages = conversation_history + current_conversation

#     try:
#         system_message = {"role": "system", "content": update_system_prompt(current_iteration, max_iterations)}
#         messages_with_system = [system_message] + messages

#         console.print(Panel("Sending request to Groq API...", style="cyan"))
#         chat_completion = groq_client.chat.completions.create(
#             messages=messages_with_system,
#             model=MAINMODEL,
#             max_tokens=8000,
#             tools=tools
#         )
#         console.print(Panel("Received response from Groq API", style="green"))

#         if chat_completion is None or not hasattr(chat_completion, 'choices') or len(chat_completion.choices) == 0:
#             raise ValueError("Received empty response from Groq API")

#         assistant_message = chat_completion.choices[0].message
        

#         if assistant_message is None:
#             raise ValueError("Received empty message content from Groq API")

#         assistant_response = assistant_message.content

#         if assistant_response is None:
#             raise ValueError("Received empty message content from Groq API")

            
#         if hasattr(chat_completion, 'usage'):
#             usage = chat_completion.usage
#             main_model_tokens['input'] += getattr(usage, 'prompt_tokens', 0)
#             main_model_tokens['output'] += getattr(usage, 'completion_tokens', 0)
#         else:
#             # Si no hay información de uso, hacemos una estimación basada en la longitud
#             main_model_tokens['input'] += len(json.dumps(messages_with_system))
#             main_model_tokens['output'] += len(assistant_response)


#         tool_calls = getattr(assistant_message, 'tool_calls', []) or []
    
#         console.print(Panel(Markdown(assistant_response), title="Groq's Response", title_align="left", border_style="blue", expand=False))

#         if tool_calls:
#             console.print(Panel("Tool calls detected", title="Tool Usage", style="bold yellow"))
#             console.print(Panel(json.dumps(tool_calls, indent=2), title="Tool Calls", style="cyan"))

#         if file_contents:
#             files_in_context = "\n".join(file_contents.keys())
#         else:
#             files_in_context = "No files in context. Read, create, or edit files to add."
#         console.print(Panel(files_in_context, title="Files in Context", title_align="left", border_style="white", expand=False))

#         for tool_call in tool_calls:
#             tool_result = await execute_tool(tool_call)
            
#             if tool_result["is_error"]:
#                 console.print(Panel(tool_result["content"], title="Tool Execution Error", style="bold red"))
#             else:
#                 console.print(Panel(tool_result["content"], title_align="left", title="Tool Result", style="green"))

#             current_conversation.append({
#                 "role": "assistant",
#                 "content": None,
#                 "tool_calls": [tool_call]
#             })

#             current_conversation.append({
#                 "role": "tool",
#                 "content": tool_result["content"],
#                 "tool_call_id": getattr(tool_call, 'id', 'unknown_id')
#             })

#         messages = conversation_history + current_conversation

#         try:
#             tool_response = groq_client.chat.completions.create(
#                 messages=messages,
#                 model=TOOLCHECKERMODEL,
#                 max_tokens=8000,
#                 tools=tools
#             )
#             if hasattr(tool_response, 'usage'):
#                 tool_checker_tokens['input'] += getattr(tool_response.usage, 'prompt_tokens', 0)
#                 tool_checker_tokens['output'] += getattr(tool_response.usage, 'completion_tokens', 0)

#             tool_checker_response = tool_response.choices[0].message.content if tool_response.choices else None
#             if tool_checker_response:
#                 console.print(Panel(Markdown(tool_checker_response), title="Groq's Response to Tool Result",  title_align="left", border_style="blue", expand=False))
#                 assistant_response += "\n\n" + tool_checker_response
#         except Exception as e:
#             error_message = f"Error in tool response: {str(e)}"
#             console.print(Panel(error_message, title="Error", style="bold red"))
#             assistant_response += f"\n\n{error_message}"

#     except Exception as e:
#         console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
#         return "I'm sorry, there was an error communicating with the AI. Please try again.", False

#     if assistant_response:
#         current_conversation.append({"role": "assistant", "content": assistant_response})

#     conversation_history = messages + [{"role": "assistant", "content": assistant_response}]


#     return assistant_response, CONTINUATION_EXIT_PHRASE in assistant_response


async def chat_with_groq(user_input, image_path=None, current_iteration=None, max_iterations=None):
    global conversation_history, main_model_tokens

    current_conversation = []

    if image_path:
        image_base64 = encode_image_to_base64(image_path)
        if image_base64.startswith("Error"):
            console.print(Panel(f"Error encoding image: {image_base64}", title="Error", style="bold red"))
            return "I'm sorry, there was an error processing the image. Please try again.", False

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        current_conversation.append(image_message)
    else:
        current_conversation.append({"role": "user", "content": user_input})

    messages = conversation_history + current_conversation

    try:
        system_message = {"role": "system", "content": update_system_prompt(current_iteration, max_iterations)}
        messages_with_system = [system_message] + messages

        console.print(Panel("Sending request to Groq API...", style="cyan"))
        chat_completion = groq_client.chat.completions.create(
            messages=messages_with_system,
            model=MAINMODEL,
            max_tokens=8000,
            tools=tools,
            tool_choice="auto"
        )
        console.print(Panel("Received response from Groq API", style="green"))

        if chat_completion is None or not hasattr(chat_completion, 'choices') or len(chat_completion.choices) == 0:
            raise ValueError("Received empty response from Groq API")

        assistant_message = chat_completion.choices[0].message

        if assistant_message is None:
            raise ValueError("Received empty message content from Groq API")

        assistant_response = assistant_message.content

        if assistant_response is None:
            raise ValueError("Received empty message content from Groq API")

        if hasattr(chat_completion, 'usage'):
            usage = chat_completion.usage
            main_model_tokens['input'] += getattr(usage, 'prompt_tokens', 0)
            main_model_tokens['output'] += getattr(usage, 'completion_tokens', 0)
        else:
            main_model_tokens['input'] += len(json.dumps(messages_with_system))
            main_model_tokens['output'] += len(assistant_response)

        console.print(Panel(Markdown(assistant_response), title="Groq's Response", title_align="left", border_style="blue", expand=False))

        # Procesar herramientas iniciales
        await process_tool_calls(assistant_response, current_conversation)

        # Procesar herramientas adicionales basadas en la respuesta a las herramientas
        max_tool_iterations = 5  # Prevenir bucles infinitos
        for _ in range(max_tool_iterations):
            try:
                tool_response = groq_client.chat.completions.create(
                    messages=conversation_history + current_conversation,
                    model=TOOLCHECKERMODEL,
                    max_tokens=8000,
                    tools=tools
                )
                if hasattr(tool_response, 'usage'):
                    tool_checker_tokens['input'] += getattr(tool_response.usage, 'prompt_tokens', 0)
                    tool_checker_tokens['output'] += getattr(tool_response.usage, 'completion_tokens', 0)

                tool_checker_response = tool_response.choices[0].message.content if tool_response.choices else None
                if tool_checker_response:
                    console.print(Panel(Markdown(tool_checker_response), title="Groq's Response to Tool Result", title_align="left", border_style="blue", expand=False))
                    
                    # Procesar herramientas adicionales
                    new_tool_calls = await process_tool_calls(tool_checker_response, current_conversation)
                    
                    if not new_tool_calls:
                        # Si no hay nuevas llamadas a herramientas, terminamos el ciclo
                        assistant_response += "\n\n" + tool_checker_response
                        break
                else:
                    break
            except Exception as e:
                error_message = f"Error in tool response: {str(e)}"
                console.print(Panel(error_message, title="Error", style="bold red"))
                assistant_response += f"\n\n{error_message}"
                break

        if file_contents:
            files_in_context = "\n".join(file_contents.keys())
        else:
            files_in_context = "No files in context. Read, create, or edit files to add."
        console.print(Panel(files_in_context, title="Files in Context", title_align="left", border_style="white", expand=False))

    except Exception as e:
        console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
        return "I'm sorry, there was an error communicating with the AI. Please try again.", False

    if assistant_response:
        current_conversation.append({"role": "assistant", "content": assistant_response})

    conversation_history = conversation_history + current_conversation

    return assistant_response, CONTINUATION_EXIT_PHRASE in assistant_response


async def process_tool_calls(response_content, current_conversation):
    tool_calls = extract_tool_calls_from_content(response_content)
    
    if tool_calls:
        console.print(Panel("Tool calls detected", title="Tool Usage", style="bold yellow"))
        console.print(Panel(json.dumps(tool_calls, indent=2), title="Tool Calls", style="cyan"))

        for tool_call in tool_calls:
            tool_result = await execute_tool(tool_call)
            
            if tool_result["is_error"]:
                console.print(Panel(tool_result["content"], title="Tool Execution Error", style="bold red"))
            else:
                console.print(Panel(tool_result["content"], title_align="left", title="Tool Result", style="green"))

            current_conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            })

            current_conversation.append({
                "role": "tool",
                "content": tool_result["content"],
                "tool_call_id": tool_call['id']
            })

    return tool_calls
def extract_tool_calls_from_content(content):
    tool_calls = []
    tool_patterns = {
        r'create_folder\s+(\S+)': 'create_folder',
        r'create_file\s+(\S+)\s+(.+)': 'create_file',
        r'edit_and_apply\s+(\S+)\s+(.+)': 'edit_and_apply',
        r'execute_code\s+(.+)': 'execute_code',
        r'stop_process\s+(\S+)': 'stop_process',
        r'read_file\s+(\S+)': 'read_file',
        r'read_multiple_files\s+(.+)': 'read_multiple_files',
        r'list_files(\s+\S+)?': 'list_files'
    }

    for pattern, tool_name in tool_patterns.items():
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        for i, match in enumerate(matches):
            args = match.groups()
            tool_call = {
                "id": f"{tool_name}_{i}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(create_arguments_dict(tool_name, args))
                }
            }
            tool_calls.append(tool_call)

    return tool_calls

def create_arguments_dict(tool_name, args):
    if tool_name == 'create_folder':
        return {"path": args[0]}
    elif tool_name == 'create_file':
        return {"path": args[0], "content": args[1]}
    elif tool_name == 'edit_and_apply':
        return {"path": args[0], "instructions": args[1], "project_context": ""}
    elif tool_name == 'execute_code':
        return {"code": args[0]}
    elif tool_name == 'stop_process':
        return {"process_id": args[0]}
    elif tool_name == 'read_file':
        return {"path": args[0]}
    elif tool_name == 'read_multiple_files':
        return {"paths": args[0].split()}
    elif tool_name == 'list_files':
        return {"path": args[0] if args else "."}
    else:
        return {}

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {str(e)}"


def reset_code_editor_memory():
    global code_editor_memory
    code_editor_memory = []
    console.print(Panel("Code editor memory has been reset.", title="Reset", style="bold green"))

def reset_conversation():
    global conversation_history, main_model_tokens, tool_checker_tokens, code_editor_tokens, code_execution_tokens, file_contents, code_editor_files
    conversation_history = []
    main_model_tokens = {'input': 0, 'output': 0}
    tool_checker_tokens = {'input': 0, 'output': 0}
    code_editor_tokens = {'input': 0, 'output': 0}
    code_execution_tokens = {'input': 0, 'output': 0}
    file_contents = {}
    code_editor_files = set()
    reset_code_editor_memory()
    console.print(Panel("Conversation history, token counts, file contents, code editor memory, and code editor files have been reset.", title="Reset", style="bold green"))


def save_chat():
    now = datetime.datetime.now()
    filename = f"Chat_{now.strftime('%H%M')}.md"
    
    formatted_chat = "# Groq Engineer Chat Log\n\n"
    for message in conversation_history:
        if message['role'] == 'user':
            formatted_chat += f"## User\n\n{message['content']}\n\n"
        elif message['role'] == 'assistant':
            if isinstance(message['content'], str):
                formatted_chat += f"## Assistant\n\n{message['content']}\n\n"
            elif isinstance(message['content'], list):
                for content in message['content']:
                    if content['type'] == 'tool_use':
                        formatted_chat += f"### Tool Use: {content['name']}\n\n```json\n{json.dumps(content['input'], indent=2)}\n```\n\n"
                    elif content['type'] == 'text':
                        formatted_chat += f"## Assistant\n\n{content['text']}\n\n"
        elif message['role'] == 'tool':
            formatted_chat += f"### Tool Result\n\n```\n{message['content']}\n```\n\n"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_chat)
    
    return filename

# Modifica la función main
async def main():
    global automode, conversation_history, exit_automode
    console.print(Panel("Bienvenido al Chat de Groq Engineer!", title="Bienvenida", style="bold green"))
    console.print("Escribe 'salir' para terminar la conversación.")
    console.print("Escribe 'imagen' para incluir una imagen en tu mensaje.")
    console.print("Escribe 'automode [número]' para entrar en modo Autónomo con un número específico de iteraciones.")
    console.print("Escribe 'reset' para borrar el historial de la conversación.")
    console.print("Escribe 'guardar chat' para guardar la conversación en un archivo Markdown.")
    console.print("Mientras estés en automode, presiona Ctrl+C en cualquier momento para salir del bucle de automode y volver al chat regular.")

    signal.signal(signal.SIGINT, lambda signum, frame: setattr(sys.modules[__name__], 'exit_automode', True))

    while True:
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

        if user_input.lower() == 'imagen':
            image_path = (await get_user_input("Arrastra y suelta tu imagen aquí, luego presiona enter: ")).strip().replace("'", "")

            if os.path.isfile(image_path):
                user_input = await get_user_input("Tú (prompt para la imagen): ")
                response, _ = await chat_with_groq(user_input, image_path)
            else:
                console.print(Panel("Ruta de imagen inválida. Por favor, intenta de nuevo.", title="Error", style="bold red"))
                continue
        elif user_input.lower().startswith('automode'):
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
                    response, exit_continuation = await chat_with_groq(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)

                    if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                        console.print(Panel("Modo automático completado.", title="Modo Automático", style="green"))
                        automode = False
                    else:
                        console.print(Panel(f"Iteración de continuación {iteration_count + 1} completada. Presiona Ctrl+C para salir del modo automático. ", title="Modo Automático", style="yellow"))
                        user_input = "Continúa con el siguiente paso. O DETENTE diciendo 'AUTOMODE_COMPLETE' si crees que has logrado los resultados establecidos en la solicitud original."
                    iteration_count += 1

                    if iteration_count >= max_iterations:
                        console.print(Panel("Máximo de iteraciones alcanzado. Saliendo del modo automático.", title="Modo Automático", style="bold red"))
                        automode = False

                if exit_automode:
                    console.print(Panel("\nModo automático interrumpido por el usuario. Saliendo del modo automático.", title="Modo Automático", style="bold red"))
                    automode = False
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.append({"role": "assistant", "content": "Modo automático interrumpido. ¿Cómo puedo ayudarte más?"})

            except Exception as e:
                console.print(Panel(f"\nError en modo automático: {str(e)}", title="Error de Modo Automático", style="bold red"))
                automode = False

            console.print(Panel("Salido del modo automático. Volviendo al chat regular.", style="green"))
        else:
            response, _ = await chat_with_groq(user_input)

if __name__ == "__main__":
    asyncio.run(main())