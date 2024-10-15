from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np 
import uvicorn
import sklearn
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


# Creamos la app FastAPI
app = FastAPI(title="API de Recomendación de Juegos de la plataforma STEAM",
              description="API para analizar juegos, desarrolladores y recomendaciones de la plataforma STEAM")

# Cargamos los datasets

# Cargamos los dataset
funcion_1 = pd.read_parquet('Dataset_funcion_1.parquet', engine='pyarrow')
funcion_2 = pd.read_parquet('Dataset_funcion_2.parquet', engine='pyarrow')
funcion_3 = pd.read_parquet('Dataset_funcion_3.parquet', engine='pyarrow')
funcion_4 = pd.read_parquet('Dataset_funcion_4.parquet', engine='pyarrow')
funcion_5 = pd.read_parquet('Dataset_Funcion_5.parquet', engine='pyarrow')
Data_sample = pd.read_parquet('Data_Sample_Recommendation', engine='pyarrow')


# Iniciamos la API
@app.get('/')
async def index():
    return 'Hola! Bienvenido a la API de recomedación de videojuegos de la plataforma STEAM.'

# Funcion 1
@app.get('/developer/{desarrollador}')
async def developer(desarrollador: str):
    '''
    Esta funcion devuelve la cantidad de items y porcentaje de contenido Free por año según empresa desarrolladora. 
    
    Args : desarrollador (str) Nombre del Desarrollador.

    Returns:
        Dataframe con las siguientes columnas:
            - Año: Contexto de tiempo
            - Cantidad de items: Cantidad de videojuegos
            - Contenido Free: Cantidad de juegos gratuitos
    '''
    # Primero filtramos el DataFrame por el desarrollador
    df_dev = funcion_1.loc[funcion_1['Developer'] == desarrollador]

    if df_dev.empty:
        return {"error": f"No se encontraron datos para el desarrollador {desarrollador}"}

    # Vemos de que la columna Price sea numérica usando .loc
    df_dev.loc[:, 'Price'] = pd.to_numeric(df_dev['Price'], errors='coerce')

    # Creamos la columna 'Free' donde 1 indica que el juego es gratis
    df_dev.loc[:, 'Free'] = df_dev['Price'].apply(lambda x: 1 if x == 0 else 0)

    # Agrupamos por 'Year' para calcular cantidad de juegos y porcentaje de juegos gratuitos
    result = df_dev.groupby('Year').agg(
        Cantidad_Items=('Id', 'count'),
        Contenido_Free=('Free', lambda x: f"{(x.mean() * 100):.0f}%")
    ).reset_index()

    # Convertimos el DataFrame result a una lista de diccionarios
    return result.to_dict(orient='records')

# Funcion 2
@app.get('/userdata/{User_id}')
async def userdata(User_id: str):
    '''
    Esta función devuelve información sobre un usuario según su 'user_id'.

    Args:
        User_id (str): Identificador único del usuario.

    Returns:
        dict: Un diccionario que contiene información sobre el usuario.
            - 'cantidad_dinero' (float): Cantidad de dinero gastado por el usuario.
            - 'porcentaje_recomendacion' (float): Porcentaje de recomendaciones realizadas por el usuario.
            - 'total_items' (int): Cantidad de items que tiene el usuario.
    '''
    # Obtenemos solamente la informacion relacionada con el usuario
    user_data = funcion_2[funcion_2['User_Id'] == User_id]
    
    if user_data.empty:
        return {"error": f"No se encontraron datos para el usuario {User_id}"}
    
    if 'Price' not in user_data.columns or 'Items_Count' not in user_data.columns:
        return {"error": "Faltan columnas requeridas en los datos del usuario."}
    
    # Cantidad de dinero gastado
    dinero_gastado = float(user_data['Price'].sum().round(2))

    # Porcentaje de recomendación
    total_reviews = len(user_data)
    recomendaciones = user_data[user_data['Recommend'] == True]
    porcentaje_recomendacion = (len(recomendaciones) / total_reviews * 100) if total_reviews > 0 else 0.0

    # Cantidad de items 
    cantidad_items = int(user_data['Items_Count'].iloc[0])  # Asegúrate de que esta columna existe

    return {
        "Usuario": User_id,
        "Dinero gastado": f"{dinero_gastado:.2f} USD",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "Cantidad de items": cantidad_items
    }

# Funcion 3
@app.get('/userforgenre/{genero}')
async def UserForGenre(genero: str):
    '''
    Esta función devuelve información sobre el usuario con más horas de juego según el género de sus items.

    Args:
        genero (str): Nombre del género.

    Returns:
        dict: Un diccionario que contiene información sobre el usuario con más horas de juego por año y género.
            - 'User Id' (int): Identificador único del usuario.
            - 'Year' (int): Año.
            - 'Playtime_Forever' (float): Número de horas jugadas.
    '''    
    # Explode los géneros en el DataFrame
    df_exploded = funcion_3.explode('Genres')
    
    # Filtramos por el género dado
    genre_filtered = df_exploded[df_exploded['Genres'].apply(lambda x: genero in x)]
    
    if genre_filtered.empty:
        return f"No hay datos para el género {genero}"

    # Agrupamos por User_Id y Year, y acumulamos horas jugadas
    hours_per_year = genre_filtered.groupby(['User_Id', 'Year'])['Playtime_Forever'].sum().reset_index()

    # Sumamos horas totales por usuario
    total_hours = hours_per_year.groupby('User_Id')['Playtime_Forever'].sum().reset_index()

    # Encontramos el usuario con más horas
    top_user = total_hours.loc[total_hours['Playtime_Forever'].idxmax()]

    # Creamos la lista de horas por año
    hours_by_year = hours_per_year[hours_per_year['User_Id'] == top_user['User_Id']].to_dict(orient='records')

    return {
        "Usuario con más horas jugadas para Género": top_user['User_Id'],
        "Horas jugadas": hours_by_year
    }

# Funcion 4
@app.get('/best_developer_year/{año}')
async def best_developer_year(year: int):
    # Filtramos por año y condiciones adicionales

    if year not in funcion_4["Year"].unique():
        return f"No se tienen registros del año {year}"
        
    filtered_year = funcion_4[(funcion_4["Year"] == year) & 
                              (funcion_4["Recommend"] == True) & 
                              (funcion_4["Analisis_Sentimiento"] == 2)]
    
    # Agrupamos por desarrollador y contar la cantidad de juegos recomendados
    grouped_dev = filtered_year.groupby("Developer")["App_Name"].count().sort_values(ascending=False)
    
    # Retorna los 3 mejores desarrolladores
    top_3 = grouped_dev.head(3)
    
    # Obtenemos los resultados en un formato amigable
    return [
        {"Puesto 1": {"Developer": top_3.index[0]}},
        {"Puesto 2": {"Developer": top_3.index[1]}},
        {"Puesto 3": {"Developer": top_3.index[2]}}
    ]

# Funcion 5
@app.get('/developer_reviews_analysis/{desarrolladora}')
async def developer_reviews_analysis( desarrollador : str):
    '''
    Esta funcion devuelve un diccionario en el cual la clave es el desarrollador y su valor es una lista 
    en donde se detallan la cantidad reseñas de caracter positivo y negativo

    Args:
        desarrollador (str): Nombre del desarrollador.

    Returns:
        dict: Un diccionario que contiene información sobre el desarrollador y la cantidad de reseñas positivas y negativas acerca de él.
            - Desarrolador (int): Nombre de la empresa desarrolladora.
            - 'Negative' (int): Cantidad de reseñas negativas.
            - 'Positive' (float): Cantidad de reseñas negativas.

    '''
    filtered_dev = funcion_5[funcion_5["Developer"] == desarrollador]

    if filtered_dev.empty:
        return f"No se encontraron datos para el desarrollador {desarrollador}"
    
    Positive_filter = filtered_dev[filtered_dev["Analisis_Sentimiento"] == 2]
    Negative_filter = filtered_dev[filtered_dev["Analisis_Sentimiento"] == 0]

    reviews = {
        "Negative": Negative_filter.shape[0],  
        "Positive": Positive_filter.shape[0]
    }

    resultado = {desarrollador: reviews}

    return resultado

# Modelo de recomendacion
@app.get('/recomendacion_usuario/{user_id}')
async def recomendacion_usuario(user_id: str):
    if user_id not in Data_sample['User_Id'].values:
        raise HTTPException(status_code=404, detail=f"No se encontró el usuario con ID: {user_id}")

    # Preprocesamiento
    tfidf = TfidfVectorizer(stop_words='english')
    Data_sample['App_Name'] = Data_sample['App_Name'].fillna('')
    tfidf_matrix = tfidf.fit_transform(Data_sample['App_Name'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    matching_users = Data_sample[Data_sample['User_Id'] == user_id]
    if not matching_users.empty:
        user_index = matching_users.index[0]
        recommendations = []
        seen_games = set()

        for i, score in sorted(enumerate(cosine_sim[user_index]), key=lambda x: x[1], reverse=True):
            if Data_sample['Recommend'][i] and Data_sample['Analisis_Sentimiento'][i] in [0, 1, 2]:
                app_name = Data_sample['App_Name'][i]
                if app_name not in seen_games:
                    recommendations.append({"App_Name": app_name, "Similaridad": score})
                    seen_games.add(app_name)

        top_recommendations = recommendations[:5]

        if not top_recommendations:
            raise HTTPException(status_code=404, detail="No se encontraron recomendaciones.")

        return {"Top_Recomendaciones": top_recommendations}
    else:
        raise HTTPException(status_code=404, detail="No se encontró el usuario con ID: {user_id}")
