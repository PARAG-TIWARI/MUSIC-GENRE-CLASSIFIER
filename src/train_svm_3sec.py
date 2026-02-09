df = pd.read_csv("dataset/features_3_sec.csv")

X = df.drop(["filename", "label", "length"], axis=1)
y = df["label"]
