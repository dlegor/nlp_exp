import streamlit as st
import spacy
from spacy import displacy
import pandas as pd
from PIL import Image
from SessionState import get
from typing import Any

#Text Examples
TEXT1="Durante el Ejercicio Fiscal 2013, se invirtieron mas recursos a este tipo de obras."
TEXT2="Si bien se desarrollaron acciones, también es importante destacar que en algunos casos se desarrollaron las gestiones pero el evento se pospuso por motivos de la contingencia por COVID-19."
TEXT3="Al tratarse de un programa de conducción e instrumentación de política, está directamente relacionado con la redefinición que se está sucediendo de la Política Nacional de Vivienda, debido al cambio de administración. Esto ha provocado que existan diferencias entre las metas programadas a inicio de año y los avances registrados en cada unos de los semestres."

#Lenguages Models
SPACY_MODEL_NAMES = ["es_core_news_sm", "es_core_news_md","es_core_news_lg"]
HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

@st.cache(allow_output_mutation=True)
def load_model(name):
    return spacy.load(name)
@st.cache(allow_output_mutation=True)
def process_text(model_name, text):
    nlp = load_model(model_name)
    return nlp(text)

@st.cache()
def load_image():
    try:
        return Image.open('images/nlp.jpeg')
    except:
        return Image.open('app/images/nlp.jpeg')



def statistic_text(text_input:str,nlp)->Any:
    num_char=len(text_input)
    doc=nlp(text_input)
    Nnoun_chunks=len([l for l in doc.noun_chunks])
    NToken=len([l.text for l in doc])
    NVerbs=len([l for l in doc if 'VERB' in l.tag_])
    SinStopWords=len([token.text for token in doc if not token.is_stop])
    return num_char,Nnoun_chunks,NToken,NVerbs,SinStopWords

def documentation():
    st.markdown("""
    
    If you want to know more about it, check the links:.\n\n

    """)

    st.sidebar.markdown("""
    Select the topics for more information.
    """)
    a1=st.sidebar.checkbox(label='Text Mining',value=True)
    a2=st.sidebar.checkbox(label='Natural language processing',value=True)
    a3=st.sidebar.checkbox(label='Language Model',value=True)
    a4=st.sidebar.checkbox(label='POS tagging')
    a5=st.sidebar.checkbox(label='Chunks')
    a6=st.sidebar.checkbox(label='Token')
    a7=st.sidebar.checkbox(label='NER')
    a8=st.sidebar.checkbox(label='Vector Representation and Embeddings')

    if a1:
        st.markdown("""\n\n

        Text Mining:
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Miner%C3%ADa_de_textos#:~:text=La%20miner%C3%ADa%20de%20textos%20es,est%C3%A1%20expl%C3%ADcita%20dentro%20del%20texto.)
        * [Book: Text Mining with R ](https://www.tidytextmining.com/)
        
        \n""")
    if a2:
        st.markdown("""
        Natural language processing
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Procesamiento_de_lenguajes_naturales)
        * [Book: Natural Lenguage Processing with Pytho](https://www.nltk.org/book/)
        
        """)
    if a3:
        st.markdown("""
        Language Model
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Modelaci%C3%B3n_del_lenguaje)
        * [Book: Speech and Lenguage Processing](http://web.stanford.edu/~jurafsky/slp3/3.pdf)
        
        """)
    if a4:
        st.markdown("""
        Part-of-speech tagging(POS-tagging) 
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Etiquetado_gramatical)
        * [Book: Speech and Lenguage Processing](http://web.stanford.edu/~jurafsky/slp3/8.pdf)
        
        """)
    if a5:
        st.markdown("""
        Chunks 
        
        * [Wikipedia](https://en.wikipedia.org/wiki/Phrase_chunking)
        * [Book: Natural Lenguage Processing with Pytho ](http://www.nltk.org/book/ch07.html)
        
        """)
    if a6:
        st.markdown("""
        Token 
        
        * [Wikipedia](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
        * [Book: Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)
        
        """)
    if a7:
        st.markdown("""
        Named-entity recognition
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Reconocimiento_de_entidades_nombradas)
        * [Course: From Languages to Information](https://web.stanford.edu/class/cs124/lec/Information_Extraction_and_Named_Entity_Recognition.pdf)
        
        """)

    if a8:
        st.markdown("""
        Vector Representation y Embedding
        
        
        * [Wikipedia](https://es.wikipedia.org/wiki/Modelo_de_espacio_vectorial)
        * [Wikipedia: Embedding](https://es.wikipedia.org/wiki/Word_embedding)
        * [Book: Speech and Lenguage Processing](http://web.stanford.edu/~jurafsky/slp3/6.pdf)
        
        """)

def main():
    " Main function for the app to work"
    st.sidebar.title("Natural Language Processing")
    t1=st.sidebar.selectbox(label="Select Text",options=['Text 1','Text 2','Text 3'])


    spacy_model = st.sidebar.selectbox("Language Model", SPACY_MODEL_NAMES,index=1)
    model_load_state = st.info("Uploading Language model {}".format(spacy_model))
    nlp = load_model(spacy_model)
    model_load_state.empty()

    if t1=='Text 1':
        text = st.text_area("Input Text", TEXT1)
        st.text("You can edit the text above.")
    elif t1=='Text 2':
        text = st.text_area("Input Text", TEXT2)
        st.text("You can edit the text above..")
    elif t1=='Text 3':
        text = st.text_area("Input Text", TEXT3)
        st.text("You can edit the text above..")
    else:
        st.info("Error")

    doc = process_text(spacy_model, text)

    #if "parser" in nlp.pipe_names:
        #st.header("Análisis del Texto & Part-of-speech tags")
        # st.sidebar.header("Relación entre palabras")
        # split_sents = st.sidebar.checkbox("División de oraciones", value=True)
        # collapse_punct = st.sidebar.checkbox("Colapso de puntuación", value=True)
        # collapse_phrases = st.sidebar.checkbox("Colapso de frases",value=True)
        # dependencies=st.sidebar.checkbox("Dependencias")
        # compact = st.sidebar.checkbox("Modo Compacto")
        # options = {
        # "collapse_punct": collapse_punct,
        # "collapse_phrases": collapse_phrases,
        # "compact": compact,
        # "dependecies":dependencies,
        # "bg":"#121112",
        # "color":"#ffffff",
        # "font":"IBM Plex Sans"}
        # docs = [span.as_doc() for span in doc.sents] if split_sents else [doc]
        # for sent in docs:
        #     html = displacy.render(sent, options=options,style='dep')
        #     # Double newlines seem to mess with the rendering
        #     html = html.replace("\n\n", "\n")
       
        # if split_sents and len(docs) > 1:
        #        st.markdown("> {}".format(sent.text))
        # st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        # st.text("Desplazate a la derecha la barra gris para ver toda la imagen.")
        # st.sidebar.info("Si no se visualiza la imagen,prueba una a una las opciones.")

    if "ner" in nlp.pipe_names:
        st.header("NER")
        st.sidebar.header("NER in the Model")
    
         #default_labels = ["PERSON", "ORG", "GPE", "LOC"]
        default_labels =nlp.pipe_labels['ner']
        labels = st.sidebar.multiselect("NER type", nlp.get_pipe("ner").labels, default_labels)
        html = displacy.render(doc, style="ent", options={"ents": labels})
        # Newlines seem to mess with the rendering
        html = html.replace("\n", " ")
        st.write(HTML_WRAPPER.format(html), unsafe_allow_html=True)
        attrs = ["text", "label_", "start", "end", "start_char", "end_char"]
        if "entity_linker" in nlp.pipe_names:
            attrs.append("kb_id_")
        data = [
            [str(getattr(ent, attr)) for attr in attrs]
            for ent in doc.ents
            if ent.label_ in labels
            ]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)


#    if "textcat" in nlp.pipe_names:
#        st.header("Text Classification")
#        st.markdown("> {}".format(text))
 #       df = pd.DataFrame(doc.cats.items(), columns=("Label", "Score"))
 #       st.dataframe(df)

    st.header("Tokens and  Attributes")

    if st.button("Show Attributes"):
        attrs = [
            "idx",
            "text",
            "lemma_",
            "pos_",
            "tag_",
            "dep_",
            "head",
            "ent_type_",
            "ent_iob_",
            "shape_",
            "is_alpha",
            "is_ascii",
            "is_digit",
            "is_punct",
            "like_num"]
        data = [[str(getattr(token, attr)) for attr in attrs] for token in doc]
        df = pd.DataFrame(data, columns=attrs)
        st.dataframe(df)


    st.header("Vector")
    vector_size = nlp.meta.get("vectors", {}).get("width", 0)
    if vector_size:
        st.subheader("Vectors and Similarity")
        #st.code(nlp.meta["vectors"])
        text1 = st.text_input("Sentence or Word 1", text)
        text2 = st.text_input("Sentence o Word 2", "Goberno")
        doc1 = process_text(spacy_model, text1)
        doc2 = process_text(spacy_model, text2)
        similarity = doc1.similarity(doc2)
        if similarity > 0.5:
            st.success(similarity)
        else:
            st.error(similarity)
        st.text("You can edit the sentence text.")

        
    st.sidebar.header("Statistics")

    if st.sidebar.checkbox(label="Show",value=True):
        st.subheader("Statistics")
        num_char,Nnoun_chunks,NToken,NVerbs,SinStopWords=statistic_text(text_input=text,nlp=nlp)
        Dic_Stat={'Sentence Chars':num_char,
        'Num Chunks':Nnoun_chunks,
        'Num Tokens': NToken,
        'Num Verbs':NVerbs,
        'Num StopWords': SinStopWords}
        st.write(Dic_Stat)




session_state = get(password='')

if session_state.password != "Xl8#deFG?dg":
    pwd_placeholder = st.sidebar.empty()
    pwd = pwd_placeholder.text_input("Password:", value="", type="password")
    session_state.password = pwd
    if session_state.password == "Xl8#deFG?dg":
        pwd_placeholder.empty()
        st.image(load_image())
        st.title("App Demo NLP Techniques")
        optiones=st.sidebar.selectbox(label='Options',options=['Demo','Documentation'])
        if optiones=='Demo':
            main()
        elif optiones=='Documentation':
            documentation()

    else:
        st.error("the password you entered is incorrect")
        st.stop()
else:
    st.image(load_image())
    st.title("App Demo NLP Techniques")
    optiones=st.sidebar.selectbox(label='Options',options=['Demo','Documentation'])
    if optiones=='Demo':
        main()
    elif optiones=='Documentation':
        documentation()

