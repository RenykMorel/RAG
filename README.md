# RAG
Guía de Integración del Sistema RAG con Flask y LLMs para BotPro
Índice
1.	Introducción
2.	Requisitos previos
3.	Estructura del proyecto
4.	Instalación del entorno
5.	API y endpoints
6.	Ejecución del sistema
7.	Optimización y resolución de problemas
8.	Ejemplos de uso
Introducción
Este documento proporciona una guía detallada para implementar un sistema RAG (Retrieval-Augmented Generation) usando Flask como backend y un frontend web sencillo. El sistema permite cargar documentos PDF, procesarlos utilizando técnicas de embedding y vectorización, y luego realizar consultas sobre estos documentos utilizando modelos de lenguaje Transformer HGF.
El sistema incluye las siguientes funcionalidades:
•	Carga y procesamiento de documentos PDF
•	Creación de una base de datos vectorial para búsqueda semántica
•	Interfaz de chat para consultas en lenguaje natural
•	Visualización de fuentes y contexto utilizados para las respuestas
•	Traducción automática entre idiomas
•	Exportación de conversaciones
Requisitos previos
Hardware recomendado
•	CPU: 4 núcleos o más
•	RAM: Mínimo 8GB, recomendado 16GB+
•	Almacenamiento: 5GB libres para la aplicación y dependencias
•	GPU: Opcional pero recomendada para mejor rendimiento (CUDA compatible) NOTA: Esto es muy importante considerar.
Software necesario
•	Python 3.9+ instalado
•	Pip (gestor de paquetes de Python)
•	Git (opcional, para clonar el repositorio)
•	Navegador web moderno
Estructura del proyecto
RAG/
│
├── .venv/                   # Entorno virtual de Python
├── static/                  # Archivos estáticos para el frontend
│   ├── css/                 # Estilos CSS
│   │   └── style.css        # Archivo principal de estilos
│   └── js/                  # JavaScript
│       └── script.js        # Script del cliente
│
├── templates/               # Plantillas HTML
│   └── index.html           # Página principal
│
├── uploads/                 # Carpeta para almacenar los PDFs subidos
├── utils/                   # Utilidades y funciones auxiliares
├── app.py                   # Aplicación principal Flask
├── requirements.txt         # Dependencias del proyecto
└── .env                     # Variables de entorno (token de API, etc.) (Es importante crear unas venv para las dependencias venv_googletrans y venv_langsmith) esto optimiza recursos para evitar dependencias mas pesadas y problemáticas.
Instalación del entorno
Paso 1: Crear y activar un entorno virtual
Clonar el repositorio RAG
Git clone + la URL del repositorio
# Crear un entorno virtual
python -m venv .venv
Nota: en los archivos se encontraran con un venv llamado .venv deben eliminarlo y crear otro venv para que se instalen allí las dependencias en el usuario local este paso no se debe obviar.

# Activar el entorno virtual
# En Windows:
.venv\Scripts\activate

# En macOS/Linux:
source .venv/bin/actívate

Paso 2: Instalar dependencias
Crea un archivo requirements.txt con el siguiente contenido:
Flask==2.2.3 
Flask-Cors==3.0.10 
langchain==0.1.6 
langchain-community==0.0.16 
langchain-huggingface==0.0.3 
faiss-cpu==1.7.4 
pypdf==3.17.1 
transformers==4.36.2 
sentence-transformers==2.2.2 
torch==2.1.2 
huggingface_hub==0.20.2 
langdetect==1.0.9
googletrans==4.0.0-rc1
Instala las dependencias:
Ejecuta en la terminar: pip install -r requirements.txt

Importante: en los archivos ya hay un archivo requirements.txt se puede instalar ese en ese caso no seria necesario crear uno, pero si se va a integrar algo más seria agregarlo al requirements.txt nuevo e instalar las dependencias luego de tenerlas completas en el archivo requirements.txt
Paso 3: Configurar el token de Hugging Face
1.	Regístrate en Hugging Face si aún no lo has hecho
2.	Ve a tu perfil > Settings > Access Tokens
3.	Crea un nuevo token con permisos de lectura
4.	Crea un archivo .env en la raíz del proyecto con el siguiente contenido:
HF_TOKEN=tu_token_de_hugging_face
Aspectos importantes del backend:
1.	Procesamiento de documentos:
o	Los documentos PDF se cargan utilizando PyPDFLoader
o	Se dividen en fragmentos más pequeños con RecursiveCharacterTextSplitter
o	Se crean embeddings para cada fragmento con HGFEmbeddings
o	Se almacenan en una base de datos vectorial con FAISS (Tener cuidado con dejar fuera esta dependencia porque tiene a ser un poco compleja de instalar porque es compatible con pocas dependencias, Consejos instalarla de primero si empieza a dar incompatibilidad) 
2.	Motor de chat RAG:
o	Utiliza ConversationalRetrievalChain de LangChain
o	Recupera los fragmentos más relevantes con el retriever de FAISS
o	Genera respuestas usando modelos de Transformer de HGF
o	Mantiene el historial de conversación con ConversationBufferMemory
Por esto es necesario buena memoria para correr el modelo, no una bestialidad de memoria, pero al menos 8 sin mucho compromiso, un sobresalto de memoria puede detener el modelo.
3.	Traducción:
o	Detección automática de idioma con langdetect
o	Traducción al español con googletrans si el modelo lo usara algún cliente en ingles eliminar el venv y las dependencias de googletrans para que trabaje solo en inglés. 
o	Sistema de respaldo para traducciones con mapeo predefinido

API y endpoints
Endpoints principales:
1.	/api/upload (POST)
o	Recibe archivos PDF
o	Procesa los documentos y crea la base de datos vectorial
o	Retorna confirmación o error
2.	/api/initialize_model (POST)
o	Configura el modelo de lenguaje con los parámetros seleccionados
o	Inicializa la cadena de RAG
o	Retorna confirmación o error
3.	/api/chat (POST)
o	Recibe preguntas del usuario
o	Genera respuestas basadas en el contenido de los documentos
o	Retorna la respuesta y las fuentes utilizadas
4.	/api/pdf/<filename> (GET)
o	Sirve los archivos PDF para visualización
o	Permite la vista previa de documentos
5.	/api/summarize/<filename> (GET)
o	Genera un resumen automático del documento
o	Versión básica implementada, expandible con LLMs
Ejecución del sistema
Paso 1: Iniciar el servidor Flask
# Asegúrate de que el entorno virtual está activado se debe mostrar el (.venv) para indicar que esta en el entorno virtual, si no esta entro del entorno virtual no instalar las dependencias.
 
# Establece la variable de entorno para el token de HGF
# En Windows:
set HF_TOKEN=tu_token_de_hugging_face
# En macOS/Linux:
export HF_TOKEN=tu_token_de_hugging_face
Estos Token lo conectan a un LLM en HGF pero si desean descargarlos no hay problemas lo pueden usar en local los modelos de código Open, pero el modelo usa transformer para que sea más optimo y no consuma tantos recuersos.

# Inicia el servidor Flask
python app.py
El servidor se ejecutará en http://localhost:5000 por defecto.
Paso 2: Acceso a la interfaz web
1.	Abre tu navegador y accede a http://localhost:5000
2.	Sigue los pasos que se muestran en la interfaz: 
o	Sube uno o más documentos PDF
o	Crea la base de datos vectorial
o	Configura e inicializa el modelo
o	Comienza a hacer preguntas sobre los documentos
Flujo de trabajo típico:
1.	Carga de documentos:
o	Arrastra y suelta archivos PDF en la zona designada
o	Haz clic en "Crear base de datos vectorial"
o	Espera a que se procesen los documentos
2.	Configuración del modelo:
o	Selecciona el modelo (Llama-3-8B o Mistral-7B) Recomendación Mistral tiene mejor rendimiento, pero se puede usar cualquier LLM por defecto solo trae estos dos pero se le puede integrar las cantidades de LLM que quieran.
o	Ajusta los parámetros (temperatura, max tokens, top-k)
o	Haz clic en "Inicializar Chatbot"
3.	Consultas:
o	Escribe preguntas en el campo de chat
o	Recibe respuestas basadas en el contenido de los documentos
o	Revisa las fuentes utilizadas para generar la respuesta
o	Opcionalmente, exporta la conversación
Optimización y resolución de problemas
Optimización de rendimiento:
1.	Ajuste de parámetros de chunking:
o	Modifica chunk_size y chunk_overlap en la función load_documents según la naturaleza de tus documentos
o	Para documentos técnicos: chunks más pequeños (512-1024)
o	Para documentos narrativos: chunks más grandes (1024-2048)
2.	Selección de modelo:
o	Llama-3-8B: mejor calidad de respuestas, más recursos
o	Mistral-7B: más rápido, menor uso de recursos
3.	Ajuste de temperatura:
o	Valores bajos (0.1-0.3): respuestas más deterministas y factuales
o	Valores altos (0.7-1.0): respuestas más creativas y variadas
Problemas comunes:
1.	Error "No se encontró la variable de entorno HF_TOKEN":
o	Asegúrate de haber configurado correctamente el token de HGF
o	Verifica que la variable de entorno esté disponible para el proceso de Python
2.	Errores de memoria insuficiente:
o	Reduce el tamaño de los chunks
o	Procesa menos documentos a la vez
o	Usa un modelo más pequeño o reduce max_tokens
3.	Respuestas irrelevantes:
o	Aumenta el valor de top_k para considerar más contexto Esto es crucial para controlar la aleatoriedad y la calidad de las respuestas.
o	Reduce la temperatura para respuestas más deterministas
o	Mejora el prompt en la función initialize_qa_chain
4.	Problemas con traducciones:
o	Si googletrans falla, el sistema usa un respaldo de mapeo básico
o	Considera implementar otra API de traducción más robusta yo use esta porque no necesita muchos recursos
Ejemplos de uso
Caso de uso 1: Análisis de documentos legales
Configuración recomendada:
•	Modelo: Llama-3-8B
•	Temperatura: 0.2
•	Max tokens: 2048
•	Top-K: 5
Preguntas de ejemplo:
•	"¿Cuáles son las cláusulas principales del contrato?"
•	"Explica las condiciones de terminación del acuerdo"
•	"¿Qué responsabilidades tiene cada parte según el documento?"
Caso de uso 2: Consulta de documentación técnica
Configuración recomendada:
•	Modelo: Mistral-7B
•	Temperatura: 0.3
•	Max tokens: 1024
•	Top-K: 3
Preguntas de ejemplo:
•	"¿Cómo se configura el parámetro X según la documentación?"
•	"Explica el proceso de instalación descrito en el manual"
•	"¿Cuáles son los requisitos mínimos del sistema?"
Caso de uso 3: Análisis de informes financieros
Configuración recomendada:
•	Modelo: Llama-3-8B
•	Temperatura: 0.1
•	Max tokens: 2048
•	Top-K: 4
Preguntas de ejemplo:
•	"¿Cuál fue el ingreso neto del último trimestre?"
•	"Resume los principales riesgos mencionados en el informe"
•	"¿Cómo ha evolucionado el EBITDA en comparación con el año anterior?"
________________________________________
Conclusión
Has implementado con éxito un sistema RAG que combina búsqueda de información y generación de respuestas utilizando modelos de lenguaje. Este sistema te permite extraer información precisa de tus documentos PDF y obtener respuestas en lenguaje natural.
Para mejorar el sistema, considera:
•	Implementar más modelos de lenguaje
•	Añadir soporte para más tipos de documentos
•	Mejorar los algoritmos de chunking y embebido
•	Implementar un sistema de evaluación de respuestas
•	Añadir autenticación y multiusuario
Esto para mi es bastante básico, pero es 100% funcional, requiere mejoras y ajustes para el uso correcto que se le quiera dar, cualquier otra cosa estoy abierto, buena acogida de mi bebe.
