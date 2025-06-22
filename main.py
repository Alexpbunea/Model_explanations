import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.metrics import AUC

from deel.influenciae.common import InfluenceModel, ExactIHVP
from deel.influenciae.influence import FirstOrderInfluenceCalculator
from deel.influenciae.utils import ORDER
from deel.influenciae.trac_in import TracIn
from keras.losses import BinaryCrossentropy

import warnings
warnings.filterwarnings('ignore')

import utils

batch_size = utils.batch_size

#Reading the dataset

train_file_path = './titanic.csv' #adjust it for drive
original_df = pd.read_csv(train_file_path)
print(original_df)

df = original_df.copy()
df.rename(columns={"PassengerId": "ID"}, inplace=True)
df = df.drop(columns=["Name", "Ticket", "Cabin"])
# print(df)


# print(df.isna().sum())


df = df.dropna(subset=["Embarked"])
# print(df)


df = df.copy()
df["MissAge"] = df['Age'].isna().astype(int)
df.fillna({'Age':0}, inplace=True)
# print(df)


# print(df.isna().sum())


sex_trans = LabelEncoder()
df["Sex"] = sex_trans.fit_transform(df["Sex"])
Emb_trans = LabelEncoder()
df["Embarked"] = sex_trans.fit_transform(df["Embarked"])
# print(df)


Nomalize = StandardScaler()
Nomalize_cols = ["Age", "Fare"]
df[Nomalize_cols] = Nomalize.fit_transform(df[Nomalize_cols])
# print(df)


from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


X = df.drop(columns=["Survived"])
y = df["Survived"]

IDs = X["ID"].values.reshape(-1,1).astype(np.float32)
IDs = IDs / 1e7

X = X.drop(columns=["ID"]).values.astype(np.float32)
X = np.hstack((X, IDs))
y = to_categorical(y.values, num_classes=2)

# print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# print(len(train_ds))
# print(len(test_ds))



from keras import Sequential
from keras import layers

model = Sequential([
    layers.Dense(32, activation="relu", input_shape=(9,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(2, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()]), #run_eagerly=True)
model.fit(train_ds.batch(batch_size), epochs=10, validation_data=test_ds.batch(batch_size), verbose=2)
model.evaluate(test_ds.batch(batch_size), verbose=2)



#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#FirstOrderInfluenceCalculator

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
#TRACIN

model = Sequential([
    layers.Dense(32, activation="relu", input_shape=(9,)),
    layers.Dense(16, activation="relu"),
    layers.Dense(2,activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", AUC()]) #run_eagerly=True)

epochs = 10
unreduced_loss_fn = BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
model_list = []
model_list.append(InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss_fn))

for i in range(epochs):
  model.fit(train_ds.batch(batch_size), epochs=1, validation_data=test_ds.batch(batch_size), verbose=2)
  updated_model = tf.keras.models.clone_model(model)
  updated_model.set_weights(model.get_weights())
  model_list.append(InfluenceModel(model, start_layer=-1, loss_function=unreduced_loss_fn))

model.evaluate(test_ds.batch(batch_size), verbose=2)


influence_calculator_tracin = TracIn(model_list, 0.01)
samples_to_explain = test_ds.take(5).batch(1)
explanation_ds = influence_calculator_tracin.top_k(samples_to_explain, train_ds.batch(8), k=3, order=ORDER.DESCENDING)


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
"""
RESOLVING THE ASSIGMENT
"""
print("--------------------------------")
print("RESOLVING THE ASSIGMENT")
print("--------------------------------")

scores_dict = utils.collect_influence({'foi': influence_calculator,
                                 'tracin': influence_calculator_tracin},
                                 test_ds.take(100), train_ds, k=200)



""""
PLOT THE GRAPHS
"""
# scores_foi = np.median(np.vstack(scores_dict["foi"]), axis=0)
# scores_tracin = np.median(np.vstack(scores_dict["tracin"]), axis=0)
# #print(scores_dict)

# for method, all_lists in scores_dict.items():
#    median_curve = np.median(np.vstack(all_lists), axis=0)
#    utils.plot_long_tail(median_curve, f"Median curve {method.upper()}")
#    utils.plot_cdf(median_curve, method.upper())
#    utils.plot_method_comparison(scores_foi, scores_tracin)


"""
How do influence lists compare across the two methods?
"""
iterator = iter(influence_calculator.top_k(test_ds.take(1).batch(1),
                                           train_ds.batch(8), k=len(train_ds), order=ORDER.DESCENDING))
(_, _), scores_a, ids_a = next(iterator)

iterator = iter(influence_calculator_tracin.top_k(test_ds.take(1).batch(1),
                                           train_ds.batch(8), k=len(train_ds), order=ORDER.DESCENDING))
(_, _), scores_b, ids_b = next(iterator)



ids_a = [round(float(s[-1].numpy() * 1e7)) for s in ids_a[0]]
ids_b = [round(float(s[-1].numpy() * 1e7)) for s in ids_b[0]]


tau, spr, jac = utils.rank_similarity(ids_a, ids_b, top_k=len(train_ds))
print(f"Kendall Tau: {tau:.3f} | Spearman: {spr:.3f} | Jaccard@711: {jac:.3f}")
#if k = 50 => Kendall Tau: nan | Spearman: nan | Jaccard@50: 0.010
#if top_k = 50 => Kendall Tau: 0.000 | Spearman: 0.000 | Jaccard@50: 0.000
#if top_k = 10 => Kendall Tau: 0.000 | Spearman: 0.000 | Jaccard@10: 0.000


"""
What happens if you remove the top-k points from the training set and retrain the model? Is model performance affected?
"""
k_vals = [1, 5, 10, 25, 50, len(train_ds)]

model.compile(optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["accuracy", AUC(name="auc")]) 

baseline_loss, baseline_acc, baseline_auc = model.evaluate(test_ds.batch(batch_size), verbose=0)


for k in k_vals:
    # First-Order
    ids_drop = ids_a[:k]
    loss, acc, auc = utils.retrain_without(original_df, X_train, y_train, model, test_ds, ids_drop)
    print(f"FOI  -k={k:<2}  ΔAcc={acc-baseline_acc:+.4f}  ΔAUC={auc-baseline_auc:+.4f}")
    
    # TracIn
    ids_drop = ids_b[:k]
    loss, acc, auc = utils.retrain_without(original_df, X_train, y_train, model, test_ds, ids_drop)
    print(f"TracIn-k={k:<2}  ΔAcc={acc-baseline_acc:+.4f}  ΔAUC={auc-baseline_auc:+.4f}")






