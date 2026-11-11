import pandas as pd
from chefboost import Chefboost as chef

df = pd.read_csv("data/play_tennis_numeric.csv").drop(columns="Day")
df = df.rename(columns={"Play": "Decision"})
df = df.astype({c: object for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])})

if __name__ == "__main__":
    model = chef.fit(df, config={"algorithm": "C4.5", "enableParallelism": False},
                     target_label="Decision")
    print(chef.predict(model, param=["Sunny", 85, 85, "Weak"]))