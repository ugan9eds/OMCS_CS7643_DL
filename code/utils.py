import json 
import pandas as pd

'''
Merging two datasets together:

Kaggle dataset (https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)
All Recipes (https://archive.org/download/recipes-en-201706)

Parameters: path to kaggle dataset, path to all recipies 
Return: Pandas data frame with instructions, ingredients, title, photo_url 

'''
def combine_datasets(path_to_kaggle, path_to_allrecipies):
    
    def read_file(path):
        recipes_lines = [json.loads(line) for line in open(path, 'r')]
        return pd.DataFrame(recipes_lines)

    df_all_recipies = read_file(path_to_allrecipies)
    df_all_recipies = df_all_recipies[df_all_recipies['photo_url'] != "http://images.media-allrecipes.com/global/recipes/nophoto/nopicture-910x511.png"]
    df_all_recipies = df_all_recipies[["ingredients", "instructions", "photo_url", "title"]]
    
    df_kaggle = pd.read_csv(path_to_kaggle)
    df_kaggle['Instructions'] = df_kaggle.apply(lambda x: [x.Instructions], axis=1)
    df_kaggle.rename(columns={"Title":"title", "Cleaned_Ingredients":"ingredients", "Instructions": "instructions", "Image_Name":"photo_url"}, inplace = True)
    df_kaggle = df_kaggle[["ingredients", "instructions", "photo_url", "title"]]
    final_df = pd.concat([df_all_recipies, df_kaggle])
    
    return final_df