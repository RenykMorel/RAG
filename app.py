import os
import json
import base64
import tempfile
from datetime import datetime
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate  # Añadido la importación de PromptTemplate
from langdetect import detect
from googletrans import Translator

# Inicializar el traductor
translator = Translator()

app = Flask(__name__)
CORS(app)

# Configuración
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Token de API para Hugging Face
api_token = os.getenv("HF_TOKEN")
if not api_token:
    print("¡ADVERTENCIA! No se encontró la variable de entorno HF_TOKEN. El servicio podría no funcionar correctamente.")

# Modelos disponibles
list_llm = [
    "meta-llama/Meta-Llama-3-8B-Instruct", 
    "mistralai/Mistral-7B-Instruct-v0.2"
]
model_names = {
    "meta-llama/Meta-Llama-3-8B-Instruct": "Llama-3-8B",
    "mistralai/Mistral-7B-Instruct-v0.2": "Mistral-7B"
}

# Variables globales para almacenar el estado de la aplicación
global_vector_db = None
global_qa_chain = None

# Añadir esta función para eliminar texto duplicado
def remove_duplicated_text(text):
    """
    Detecta y elimina texto duplicado en la respuesta del modelo
    """
    # Si la respuesta es corta, no aplicar la lógica
    if len(text) < 20:
        return text
    
    # Dividir el texto a la mitad aproximadamente
    half_length = len(text) // 2
    
    # Verificar si la primera mitad es similar a la segunda mitad
    first_half = text[:half_length].strip()
    second_half = text[half_length:].strip()
    
    # Si hay duplicación similar (no necesariamente exacta)
    if first_half and second_half and (first_half in second_half or second_half in first_half):
        return first_half
    
    # Otro método: buscar frases repetidas consecutivas
    sentences = text.split('. ')
    filtered_sentences = []
    prev_sentence = ""
    
    for sentence in sentences:
        # Eliminar espacios extras y normalizar
        clean_sentence = sentence.strip()
        if clean_sentence and clean_sentence != prev_sentence:
            filtered_sentences.append(clean_sentence)
            prev_sentence = clean_sentence
    
    return '. '.join(filtered_sentences)

# Modificar la ruta de chat para aplicar la función
@app.route('/api/chat', methods=['POST'])
def chat():
    global global_qa_chain
    
    if global_qa_chain is None:
        return jsonify({"error": "Primero debe inicializar el chatbot"}), 400
    
    data = request.json
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "No se proporcionó ninguna pregunta"}), 400
    
    try:
        # Generar respuesta utilizando la cadena de QA
        response = global_qa_chain({"question": question})
        
        # Extraer la respuesta y los documentos fuente
        answer = response.get('answer', '')
        
        # Verificar SIEMPRE si la respuesta está en inglés y traducirla
        # Cambiar de condicional a traducción directa
        if detect_english_response(answer) or "the" in answer.lower():
            answer = translate_to_spanish(answer)
        
        # Refinar la respuesta para que sea específica a la pregunta
        answer = refine_answer(answer, question)
        
        # Eliminar duplicaciones en la respuesta y limpiar formato
        answer = remove_duplicated_text(answer)
        
        # NUEVA FUNCIÓN: Eliminar doble punto al final
        answer = fix_ending_punctuation(answer)
        
        source_docs = response.get('source_documents', [])
        
        sources = []
        for i, doc in enumerate(source_docs[:3]):  # Limitamos a 3 fuentes
            sources.append({
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0) + 1,
                "filename": os.path.basename(doc.metadata.get("source", "documento.pdf"))
            })
        
        return jsonify({
            "success": True,
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": f"Error al procesar la consulta: {str(e)}"}), 500

def fix_ending_punctuation(text):
    """
    Asegura que el texto tenga exactamente un signo de puntuación al final.
    Elimina espacios extras y puntos duplicados.
    """
    # Eliminar espacios al final
    text = text.rstrip()
    
    # Eliminar puntos duplicados al final
    while text.endswith(".."):
        text = text[:-1]
    
    # Eliminar combinaciones de " ."
    while text.endswith(" ."):
        text = text[:-2] + "."
    
    # Asegurar que hay exactamente un signo de puntuación al final
    if not text.endswith(('.', '!', '?')):
        text += '.'
        
    return text

# Función para procesar los documentos PDF
def load_documents(file_paths):
    loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        try:
            pages.extend(loader.load())
        except Exception as e:
            print(f"Error al cargar {loader.file_path}: {str(e)}")
    
    if not pages:
        raise ValueError("No se pudo extraer contenido de los documentos PDF")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=64
    )
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Función para crear la base de datos vectorial
def create_vector_db(doc_splits):
    embeddings = HuggingFaceEmbeddings()
    vector_db = FAISS.from_documents(doc_splits, embeddings)
    return vector_db

# Función para inicializar el modelo de LLM y la cadena de QA
# Función modificada para inicializar el modelo de LLM y la cadena de QA
def initialize_qa_chain(model_name, temperature, max_tokens, top_k, vector_db):
    # Prompt mejorado que enfatiza respuestas específicas y limitadas
    prompt_template = """
    <instrucciones>
    Eres un asistente en español especializado en responder preguntas sobre documentos PDF.
    
    REGLA IMPORTANTE: Responde ÚNICAMENTE a la pregunta específica que se te ha hecho.
    No proporciones información adicional que no haya sido solicitada explícitamente.
    No anticipes preguntas futuras ni respondas a preguntas no formuladas.
    
    Sé breve y directo. Una respuesta de 1-2 oraciones es suficiente para la mayoría de preguntas.
    
    Si la información exacta solicitada no está disponible en el contexto, di simplemente:
    "No encuentro información sobre eso en el documento."
    
    Usa SOLO la información del contexto proporcionado y responde SIEMPRE en español.
    </instrucciones>

    <contexto>
    {context}
    </contexto>

    <chat_previo>
    {chat_history}
    </chat_previo>

    <pregunta>
    {question}
    </pregunta>

    <respuesta>
    """
    
    # Crear el objeto PromptTemplate
    spanish_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )
    
    llm = HuggingFaceEndpoint(
        repo_id=model_name,
        huggingfacehub_api_token=api_token,
        temperature=temperature,
        max_new_tokens=max_tokens,
        top_k=top_k,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
        combine_docs_chain_kwargs={"prompt": spanish_prompt}
    )
    
    return qa_chain

# Añadir una función para refinar las respuestas y limitar su longitud
def refine_answer(answer, question):
    """
    Refina la respuesta para asegurar que es concisa y responde solo a la pregunta formulada
    """
    # 1. Eliminar respuestas a preguntas no formuladas
    lines = answer.split('\n')
    if len(lines) > 3:  # Si hay múltiples líneas, podría estar respondiendo a varias preguntas
        # Quedarnos solo con las primeras 1-3 líneas relevantes
        answer = '\n'.join(lines[:3])
    
    # 2. Comprobar si la respuesta contiene palabras clave que indican múltiples respuestas
    indicators = ["también", "además", "por otro lado", "asimismo", "igualmente", 
                 "adicionalmente", "incluso", "del mismo modo", "asimismo",
                 "otra información", "otros datos"]
    
    for indicator in indicators:
        if indicator in answer.lower():
            # Si encontramos un indicador, cortamos la respuesta en ese punto
            index = answer.lower().find(indicator)
            if index > 30:  # Solo cortar si ya hay suficiente texto antes
                answer = answer[:index].strip()
    
    # 3. Si la respuesta es muy larga (más de 200 caracteres), acortarla
    if len(answer) > 200:
        # Intentar cortar en un punto final
        sentences = answer.split('.')
        shortened = ''
        for sentence in sentences:
            if len(shortened) + len(sentence) < 200:
                shortened += sentence + '.'
            else:
                break
        answer = shortened
    
    # 4. NO añadir punto final aquí, delegarlo a fix_ending_punctuation
    # Eliminar esta línea:
    # if answer and not answer.endswith(('.', '!', '?')):
    #    answer += '.'
        
    return answer.strip()

# Función mejorada para detectar idioma
def detect_language(text):
    try:
        return detect(text)
    except:
        return "es"  # Por defecto, asumir español si hay error

# Función mejorada para traducir
def translate_to_spanish(text):
    try:
        # Detectar el idioma
        lang = detect_language(text)
        
        # Si no es español, traducir
        if lang != 'es':
            translation = translator.translate(text, src=lang, dest='es')
            return translation.text
        return text
    except Exception as e:
        print(f"Error en traducción: {str(e)}")
        # Si hay problemas con la API de Google Translate, usar la alternativa de mapeo
        return translate_with_mapping(text)

# Función para traducir usando mapeo predefinido (respaldo)
def translate_with_mapping(text):
    translations = {
        "The flight departs at": "El vuelo sale a las",
        "The document does not contain information about": "El documento no contiene información sobre",
        "I don't have that information": "No tengo esa información",
        "I cannot find": "No puedo encontrar",
        "I'm sorry": "Lo siento",
        "Based on the document": "Según el documento",
        "According to the document": "De acuerdo con el documento",
        "The document mentions": "El documento menciona",
        "There is no information": "No hay información",
        "The context does not provide": "El contexto no proporciona",
        "The information is not available": "La información no está disponible",
        "No information found": "No se encontró información",
        "Not mentioned in the document": "No se menciona en el documento",
        "AM": "AM",
        "PM": "PM",
        ".": ".",
        "International Airport": "Aeropuerto Internacional",
        "Intl Airport": "Aeropuerto Internacional"
    }
    
    # Reemplazar frases conocidas
    translated = text
    for eng, esp in translations.items():
        translated = translated.replace(eng, esp)
    
    # Si no se pudo traducir, proporcionar un mensaje de respaldo
    if translated == text and len(text) > 10:  # Solo si es texto sustancial y no cambió
        return "Lo siento, no puedo encontrar esa información en los documentos proporcionados."
    
    return translated

# 1. Mejora de la función de detección de inglés (línea ~414)
def detect_english_response(text):
    try:
        lang = detect_language(text)
        return lang == 'en'
    except:
        # Método mejorado con más indicadores en inglés
        english_indicators = [
            "the", "is", "are", "this", "that", "there", "flight", "departs", 
            "at", "sorry", "cannot", "find", "context", "does", "not", "provide",
            "information", "about", "document", "no", "available"
        ]
        
        # Convertir a minúsculas y dividir en palabras
        words = text.lower().split()
        
        # Contar palabras en inglés
        english_word_count = sum(1 for word in words if word in english_indicators)
        
        # Si hay al menos 2 palabras en inglés, probablemente toda la respuesta lo sea
        return english_word_count >= 2 or "does not" in text.lower() or "no information" in text.lower()

# Ruta para servir el frontend
@app.route('/')
def index():
    return render_template('index.html')

# API para subir archivos y crear la base de datos vectorial
@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({"error": "No se encontraron archivos"}), 400
    
    uploaded_files = request.files.getlist('files[]')
    file_paths = []
    
    for file in uploaded_files:
        if file.filename == '':
            continue
        
        # Verificar tipo de archivo
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"error": f"El archivo {file.filename} no es un PDF válido"}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        file_paths.append(file_path)
    
    if not file_paths:
        return jsonify({"error": "No se pudieron guardar los archivos"}), 400
    
    try:
        # Procesar documentos y crear base de datos vectorial
        doc_splits = load_documents(file_paths)
        global global_vector_db
        global_vector_db = create_vector_db(doc_splits)
        
        return jsonify({
            "success": True,
            "message": "Base de datos vectorial creada exitosamente",
            "files": [os.path.basename(file_path) for file_path in file_paths]
        })
    except Exception as e:
        return jsonify({"error": f"Error al procesar documentos: {str(e)}"}), 500

# API para inicializar el modelo de LLM
@app.route('/api/initialize_model', methods=['POST'])
def initialize_model():
    global global_vector_db, global_qa_chain
    
    if global_vector_db is None:
        return jsonify({"error": "Primero debe crear la base de datos vectorial"}), 400
    
    data = request.json
    model_name = data.get('model_name')
    temperature = float(data.get('temperature', 0.5))
    max_tokens = int(data.get('max_tokens', 2048))
    top_k = int(data.get('top_k', 3))
    
    if model_name not in list_llm:
        return jsonify({"error": "Modelo no válido"}), 400
    
    try:
        global_qa_chain = initialize_qa_chain(
            model_name, temperature, max_tokens, top_k, global_vector_db
        )
        
        return jsonify({
            "success": True,
            "message": "Chatbot inicializado exitosamente",
            "model": model_names.get(model_name, model_name)
        })
    except Exception as e:
        return jsonify({"error": f"Error al inicializar el modelo: {str(e)}"}), 500


# API para obtener una vista previa del PDF
@app.route('/api/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Archivo no encontrado"}), 404
    
    try:
        return send_file(file_path, mimetype='application/pdf')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API para resumir un documento
@app.route('/api/summarize/<filename>', methods=['GET'])
def summarize_document(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Archivo no encontrado"}), 404
    
    try:
        # En una implementación real con un modelo
        # Carga el documento
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Crear un texto para resumir (primeras páginas)
        text_to_summarize = "\n".join([page.page_content for page in pages[:3]])
        
        # Generar un resumen simulado por ahora
        summary = f"Este es un resumen automático del documento {filename}. El documento contiene información relevante que ha sido procesada por nuestro sistema de análisis de texto. Para obtener información más detallada, puede realizar preguntas específicas a través del chatbot."
        
        # TODO: En una implementación completa, aquí utilizaríamos un LLM para generar el resumen
        # prompt_template = "Genera un resumen en español del siguiente texto:\n\n{text}\n\nResumen:"
        # summary_prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        # llm = HuggingFaceEndpoint(repo_id="meta-llama/Meta-Llama-3-8B-Instruct", huggingfacehub_api_token=api_token)
        # chain = LLMChain(llm=llm, prompt=summary_prompt)
        # summary = chain.run(text=text_to_summarize)
        
        return jsonify({
            "success": True,
            "summary": summary
        })
    except Exception as e:
        return jsonify({"error": f"Error al generar el resumen: {str(e)}"}), 500

# Ruta para verificar el estado del servidor
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

# Configuración para servir el archivo HTML estático
@app.route('/index.html')
def serve_html():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)