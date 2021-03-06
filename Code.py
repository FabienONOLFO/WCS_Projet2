# IMPORT DES MODULES
from IPython.display import display

import pandas as pd
import numpy as np

from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import unicodedata


#IMPORTS DES DB
link_df_ref_traite = "https://raw.githubusercontent.com/FabienONOLFO/WCS_Projet2/main/DB_REF_TRAITE.csv"
link_df_ref = "https://raw.githubusercontent.com/FabienONOLFO/WCS_Projet2/main/DF_REF.csv"

DF_REF = pd.read_csv(link_df_ref)
DF_REF_TRAITE = pd.read_csv(link_df_ref_traite)

X = DF_REF.loc[:, DF_REF.columns!='title']

def main():
    # CREATION D'UN RESET DE REFERENCE DU DF_REF
    DF_RESET = DF_REF_TRAITE.copy()
    Film_name = input("Veuillez entrer un nom de film : ")
    while len(Film_name)<=1 :
        Film_name = str(input("Veuillez entrer un nom de film avec au moins 2 lettres: "))
        while set('[~!@#$%^&*()_+{}":;\']+$').intersection(Film_name):
            print("Merci de ne pas mettre de caractères spéciaux.")
            Film_name = input("Veuillez entrer un nom de film : ")
    
    # Si il n'y a aucune ligne qui correspond au nom du film alors on affiche Film non présent
    if len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])==0:    # Condition vérifié à part , ça fonctionne bien !
        print("Film non présent")
        main()
    # On créé une condition pour obliger d'avoir au moins 1 lettre
    else :
        DF_REF_FILTRE = DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)]

        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)

        DF_X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
        X_scaled = pd.DataFrame(X_scaled,columns=X.columns)
        distanceKNN = NearestNeighbors(n_neighbors=5).fit(X_scaled)

        DF_X_scaled['title'] = DF_REF_FILTRE['title']
        DF_X_scaled['Annee'] = DF_REF_FILTRE['startYear']

        if len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])>20:
            # On selectionne une année pour filtrer car il y a trop de films
            print("Voici les films trouvés : \n")
            display(DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True))
            filtre_annee = int(input("Merci de choisir une année approximative(+-10Ans)"))  
            annee_up = filtre_annee + 10
            annee_down = filtre_annee - 10
            DF_REF_FILTRE = DF_REF_FILTRE.loc[DF_REF_FILTRE['startYear']<=annee_up]
            DF_REF_FILTRE = DF_REF_FILTRE.loc[DF_REF_FILTRE['startYear']>=annee_down]    
            print("Voici les films trouvés : \n")
            display(DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True))
            Film_ref = int(input("\n Veuillez selectionner son index: \n"))
            neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[DF_X_scaled.index==Film_ref,X_scaled.columns.tolist()])
            print("\n Voici les films recommandés : \n")
            display(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))

        elif len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])==1:
            print("Voici le films trouvé : \n")
            display(DF_REF_FILTRE[['title','startYear','averageRating']].loc[DF_REF_FILTRE['title'].str.contains(Film_name,case=False)])
            Film_ref = DF_RESET.index[DF_RESET['title'].str.contains(Film_name,case=False)]
            neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[Film_ref,X_scaled.columns.tolist()])
            print("\n Voici les films recommandés : \n")
            display(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))  
            
        elif len(DF_RESET.loc[DF_RESET['title'].str.contains(Film_name,case=False)])<=20:
            print("Voici les films trouvés : \n")
            display(DF_RESET[['title','startYear','averageRating']].loc[DF_RESET['title'].str.contains(Film_name,case=False)].sort_values(by=['startYear'],ascending=True))
            Film_ref = int(input("\n Veuillez selectionner son index: \n"))
            neighbors = distanceKNN.kneighbors(DF_X_scaled.loc[DF_X_scaled.index==Film_ref,X_scaled.columns.tolist()])
            print("\n Voici les films recommandés : \n")
            display(DF_REF[['title','startYear','averageRating','runtimeMinutes']].iloc[neighbors[1][0]].sort_values(by=['startYear'],ascending=True))  
