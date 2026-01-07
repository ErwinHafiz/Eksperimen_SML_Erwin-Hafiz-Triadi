import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df[['2urvived','Pclass','Sex','Age','sibsp','Parch','Fare','Embarked']]
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    return df

def save_data(df, output_path):
    df.to_csv(output_path, index=False)

def main():
    df = load_data("dataset_raw/train_and_test2.csv")
    df_clean = preprocess_data(df)
    save_data(df_clean, "preprocessing/dataset_preprocessing/titanic_clean.csv")
    print("Preprocessing completed and data saved.")

if __name__ == "__main__":
    main()

