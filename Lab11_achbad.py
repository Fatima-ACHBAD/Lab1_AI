import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# Charger le modèle
model = tf.keras.models.load_model('votre_modele1.h5')

st.title('Classification d\'images avec TensorFlow/Keras')

# Interface utilisateur
uploaded_image = st.file_uploader('Téléchargez une image', type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    # Charger l'image depuis le fichier téléchargé
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)

    # Redimensionner l'image à 28x28 pixels (comme prévu par le modèle)
    image = cv2.resize(image, (28, 28))

    # Assurez-vous que l'image a 3 canaux (RVB)
    if image.shape[-1] == 1:
        # Si l'image a 1 canal, dupliquez-le pour créer 3 canaux identiques
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Normalisez l'image (assurez-vous que les valeurs sont comprises entre 0 et 1)
    image = image / 255.0

    # Préparez l'image pour la prédiction
    image = np.expand_dims(image, axis=0)

    # Faire la prédiction
    prediction = model.predict(image)

    # Afficher la classe prédite
    class_id = np.argmax(prediction)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']
    class_name = class_names[class_id]
    st.write(f'Classe prédite : {class_name}')

    # Afficher la probabilité pour chaque classe
    st.bar_chart(prediction.ravel())
