import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Reading the dataset

train_file_path = './titanic.csv' #adjust it for drive
original_df = pd.read_csv(train_file_path)
print(original_df)

df = original_df.copy()
df.rename(columns={"PassengerId": "ID"}, inplace=True)
df = df.drop(columns=["Name", "Ticket", "Cabin"])
print(df)


print(df.isna().sum())


df = df.dropna(subset=["Embarked"])
print(df)


df = df.copy()
df["MissAge"] = df['Age'].isna().astype(int)
df.fillna({'Age':0}, inplace=True)
print(df)


print(df.isna().sum())


sex_trans = LabelEncoder()
df["Sex"] = sex_trans.fit_transform(df["Sex"])
Emb_trans = LabelEncoder()
df["Embarked"] = sex_trans.fit_transform(df["Embarked"])
print(df)


Nomalize = StandardScaler()
Nomalize_cols = ["Age", "Fare"]
df[Nomalize_cols] = Nomalize.fit_transform(df[Nomalize_cols])
print(df)


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


X = df.drop(columns=["Survived"])
y = df["Survived"]

IDs = X["ID"].values.reshape(-1,1).astype(np.float32)
IDs = IDs / 1e7

X = X.drop(columns=["ID"]).values.astype(np.float32)
X = np.hstack((X, IDs))
y = to_categorical(y.values, num_classes=2)

print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
print(len(train_ds))
print(len(test_ds))



from keras import Sequential
from keras import layers

model = Sequential([
    layers.Dense(32, activation="relu", input_shape=(9,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(2, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)
model.fit(train_ds.batch(32), epochs=10, validation_data=test_ds.batch(32), verbose=2)
model.evaluate(test_ds.batch(32), verbose=2)


from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER
from deel.influenciae.trac_in import TracIn
from keras.losses import BinaryCrossentropy

import warnings
warnings.filterwarnings('ignore')



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
unreduced_loss = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
influence_model = InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss)

ihvp_calculator = ExactIHVP(influence_model, train_dataset=train_ds.shuffle(100).batch(4))
influence_calculator = FirstOrderInfluenceCalculator(influence_model, train_ds.batch(8), ihvp_calculator)

samples_to_explain = test_ds.take(5).batch(1)
explanation_ds = influence_calculator.top_k(samples_to_explain, train_ds.batch(8), k=3, order=ORDER.DESCENDING)

for (sample, label), top_k_values, top_k_samples in explanation_ds.as_numpy_iterator():
    sample_id = round(sample[0][-1] * 1e7)
    sample_original = original_df[original_df["PassengerId"] == sample_id]

    print(f"\nTest Sample ID: {sample_id}")
    print("Original Sample from DataFrame: ")
    print(sample_original[["Survived"]])

    influential_ids = [round(s[-1] * 1e7) for s in top_k_samples[0]]
    for i, (inf_id, score) in enumerate(zip(influential_ids, top_k_values[0])):
        inf_sample_original = original_df[original_df["PassengerId"] == inf_id]

        print(f"Influential Sample {i+1} -> ID: {inf_id}, Influence Score: {score}")
        print(inf_sample_original[["Survived"]])



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#TRACEIN

model = Sequential([
    layers.Dense(32, activation="relu", input_shape=(9,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(2,activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"], run_eagerly=True)

epochs = 10
unreduced_loss_fn = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
model_list = []
model_list.append(InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss_fn))

for i in range(epochs):
  model.fit(train_ds.batch(32), epochs=1, validation_data=test_ds.batch(32), verbose=2)
  updated_model = tf.keras.models.clone_model(model)
  updated_model.set_weights(model.get_weights())
  model_list.append(InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss_fn))

model.evaluate(test_ds.batch(32), verbose=2)


influence_calculator = TracIn(model_list, 0.01)
samples_to_explain = test_ds.take(5).batch(1)
explanation_ds = influence_calculator.top_k(samples_to_explain, train_ds.batch(8), k=3, order=ORDER.DESCENDING)
#explanation_ds = influence_calculator.estimate_influence_values_in_batches(samples_to_explain, train_ds)

for (sample, label), top_k_values, top_k_samples in explanation_ds.as_numpy_iterator():
    sample_id = round(sample[0][-1] * 1e7)
    sample_original = original_df[original_df["PassengerId"] == sample_id]

    print(f"\nTest Sample ID: {sample_id}")
    print("Original Sample from DataFrame: ")
    print(sample_original[["Survived"]])

    influential_ids = [round(s[-1] * 1e7) for s in top_k_samples[0]]
    for i, (inf_id, score) in enumerate(zip(influential_ids, top_k_values[0])):
        inf_sample_original = original_df[original_df["PassengerId"] == inf_id]

        print(f"Influential Sample {i+1} -> ID: {inf_id}, Influence Score: {score}")
        print(inf_sample_original[["Survived"]])