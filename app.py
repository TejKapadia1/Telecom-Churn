import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, 
                             roc_curve, roc_auc_score, mean_squared_error, r2_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title="Group PBL Dashboard", layout="wide")
st.title("Group PBL Dashboard")
st.sidebar.title("Navigation")

TABS = [
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rules",
    "Regression Insights"
]
tab = st.sidebar.radio("Select a section:", TABS)

@st.cache_data
def load_data():
    return pd.read_csv("Group PBL.csv")

def encode_and_scale(df, target=None):
    X = df.copy()
    if target is not None:
        X = X.drop(columns=[target])
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X_scaled

def preprocess_for_classification(df, target):
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    if y.dtype == 'O':
        y = LabelEncoder().fit_transform(y.astype(str))
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y

def preprocess_for_regression(df, target):
    df = df.dropna(subset=[target])
    X = df.drop(columns=[target])
    y = df[target]
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    return X, y

if tab == "Data Visualization":
    st.header("Data Visualization & Insights")
    df = load_data()
    st.subheader("1. Data Snapshot")
    st.write(df.head())

    st.subheader("2. Churn Rate vs Tenure Bucket (Line Graph)")
    if "Tenure_Bucket" in df.columns and "Churn_B" in df.columns:
        churn_series = df["Churn_B"]
        if churn_series.dtype == 'O':
            churn_series = LabelEncoder().fit_transform(churn_series.astype(str))
        churn_data = pd.DataFrame({
            "Tenure_Bucket": df["Tenure_Bucket"],
            "Churn_B": churn_series
        })
        churn_bucket_rate = churn_data.groupby("Tenure_Bucket")["Churn_B"].mean() * 100
        fig, ax = plt.subplots()
        ax.plot(churn_bucket_rate.index.astype(str), churn_bucket_rate.values, marker='o')
        ax.set_xlabel("Tenure Bucket")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate vs Tenure Bucket")
        plt.xticks(rotation=30)
        st.pyplot(fig)
        st.caption("Shows the percentage churn rate in each tenure bucket.")
    else:
        st.info("Columns 'Tenure_Bucket' and/or 'Churn_B' not found in the data.")

    st.subheader("3. Loyalty_Tier vs Churn Rate (Pie Chart)")
    if "Loyalty_Tier" in df.columns and "Churn_B" in df.columns:
        churn_series = df["Churn_B"]
        if churn_series.dtype == 'O':
            churn_series = LabelEncoder().fit_transform(churn_series.astype(str))
        churn_data = pd.DataFrame({
            "Loyalty_Tier": df["Loyalty_Tier"],
            "Churn_B": churn_series
        })
        churn_tier_rate = churn_data.groupby("Loyalty_Tier")["Churn_B"].mean() * 100
        labels = churn_tier_rate.index.astype(str)
        sizes = churn_tier_rate.values
        fig, ax = plt.subplots()
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%.1f%%', startangle=90, textprops={'fontsize': 10})
        ax.axis('equal')
        ax.set_title("Churn Rate (%) by Loyalty Tier")
        st.pyplot(fig)
        st.caption("Pie chart of churn rate for each Loyalty Tier. (Churn rate = churned/total in tier × 100)")
    else:
        st.info("Columns 'Loyalty_Tier' and/or 'Churn_B' not found in the data.")

    st.subheader("4. Correlation Matrix (Selected Metrics + Churn_B)")
    corr_cols = [
        "MonthlyCharges",
        "TotalCharges",
        "Tenure",
        "Customer_Lifetime_Value",
        "Satisfaction_Score",
        "Churn_B"
    ]
    existing_cols = [col for col in corr_cols if col in df.columns]
    missing_cols = [col for col in corr_cols if col not in df.columns]
    if len(existing_cols) >= 2:
        corr_df = df[existing_cols].copy()
        if "Churn_B" in existing_cols and not pd.api.types.is_numeric_dtype(corr_df["Churn_B"]):
            corr_df["Churn_B"] = LabelEncoder().fit_transform(corr_df["Churn_B"].astype(str))
        fig, ax = plt.subplots()
        sns.heatmap(corr_df.corr(), annot=True, ax=ax, cmap="coolwarm")
        st.pyplot(fig)
        st.caption(
            "Correlation matrix among key metrics and churn flag. " +
            ("Missing columns: " + ", ".join(missing_cols) if missing_cols else "")
        )
    else:
        st.warning(
            "Not enough of the selected columns found for correlation matrix. " +
            ("Missing columns: " + ", ".join(missing_cols) if missing_cols else "")
        )

    st.subheader("5. Pairplot (sample numeric columns)")
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) >= 2:
        st.info("Pairplot on up to 5 numeric columns.")
        fig = sns.pairplot(df[num_cols[:5]])
        st.pyplot(fig)
        plt.clf()

    st.subheader("6. Descriptive Statistics")
    st.write(df.describe())

    st.subheader("7. Value Counts for Categorical Columns")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        st.write(f"**{col}**")
        st.write(df[col].value_counts())

    st.subheader("8. Boxplot for Outlier Detection")
    if len(num_cols):
        col = st.selectbox("Select numeric column for boxplot", num_cols, key='box')
        fig, ax = plt.subplots()
        sns.boxplot(df[col], ax=ax)
        st.pyplot(fig)
        st.caption(f"Boxplot for {col}.")
    else:
        st.info("No numeric columns available.")

    st.subheader("9. Histogram for Numeric Columns")
    if len(num_cols):
        col2 = st.selectbox("Select numeric column for histogram", num_cols, key='hist')
        bins = st.slider("Number of bins", 5, 50, 20)
        fig, ax = plt.subplots()
        ax.hist(df[col2], bins=bins)
        st.pyplot(fig)
        st.caption(f"Histogram of {col2}.")

    st.subheader("10. Pivot Table (Categorical Columns)")
    if len(cat_cols) >= 2:
        idx_col = st.selectbox("Row category", cat_cols, key='rowcat')
        col_col = st.selectbox("Column category", cat_cols, key='colcat')
        pivot = pd.pivot_table(df, index=idx_col, columns=col_col, aggfunc='size', fill_value=0)
        st.write(pivot)
    else:
        st.info("Not enough categorical columns for a pivot table.")

elif tab == "Classification":
    st.header("Classification: KNN, Decision Tree, RF, GBRT")
    df = load_data()
    target = st.selectbox("Select target (label) column", df.columns, index=len(df.columns)-1)
    X, y = preprocess_for_classification(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosted": GradientBoostingClassifier()
    }
    metrics = []
    probs = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        probs[name] = y_prob
        metrics.append({
            "Model": name,
            "Train Acc": accuracy_score(y_train, model.predict(X_train)),
            "Test Acc": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        })

    st.subheader("Model Comparison Table")
    st.dataframe(pd.DataFrame(metrics).set_index("Model"))

    st.subheader("Confusion Matrix")
    model_sel = st.selectbox("Select model", list(models.keys()), key="cm")
    cm = confusion_matrix(y_test, models[model_sel].predict(X_test))
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    ax.set_title(f"{model_sel} Confusion Matrix")
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

    st.subheader("ROC Curve Comparison")
    fig, ax = plt.subplots()
    for name, model in models.items():
        if probs[name] is not None and len(np.unique(y_test)) == 2:  # Only for binary
            fpr, tpr, _ = roc_curve(y_test, probs[name])
            ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs[name]):.2f})")
    ax.plot([0,1],[0,1],"k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Predict on New Data")
    uploaded_file = st.file_uploader("Upload new data (same columns as features, without target)", type=['csv'])
    if uploaded_file:
        new_df = pd.read_csv(uploaded_file)
        try:
            new_X = encode_and_scale(new_df)
            sel_model = st.selectbox("Select model for prediction", list(models.keys()), key="pred")
            pred = models[sel_model].predict(new_X)
            res = new_df.copy()
            res['Predicted_Label'] = pred
            st.write(res)
            st.download_button("Download Predictions", res.to_csv(index=False), file_name="predictions.csv")
        except Exception as e:
            st.error("Ensure your data columns match the original data (except the target column).")
elif tab == "Clustering":
    st.header("Clustering: K-means & Personas")
    df = load_data()
    X = encode_and_scale(df)
    st.subheader("Elbow Chart (Optimal k)")
    fig, ax = plt.subplots()
    distortions = []
    K = range(2, 11)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        distortions.append(km.inertia_)
    ax.plot(K, distortions, marker='o')
    ax.set_xlabel('Number of clusters (k)')
    ax.set_ylabel('Distortion (Inertia)')
    ax.set_title('Elbow Method For Optimal k')
    st.pyplot(fig)

    st.subheader("KMeans Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    df['cluster'] = labels

    st.subheader("Customer Persona Table")
    persona = df.groupby('cluster').mean(numeric_only=True)
    st.write(persona)
    st.caption("Average feature values per cluster.")

    st.subheader("Download Data with Cluster Labels")
    st.download_button("Download Clustered Data", df.to_csv(index=False), file_name="clustered_data.csv")

elif tab == "Association Rules":
    st.header("Association Rule Mining")
    st.info("Find associations and rules in your data using Apriori.")
    df = load_data()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    cols = st.multiselect("Select columns for association mining (at least 2 categorical)", cat_cols, default=cat_cols[:2])
    if len(cols) < 2:
        st.warning("Select at least two categorical columns.")
    else:
        onehot = pd.get_dummies(df[cols])
        min_support = st.slider("Minimum support", 0.01, 0.5, 0.1, 0.01)
        min_confidence = st.slider("Minimum confidence", 0.1, 1.0, 0.5, 0.05)
        freq_items = apriori(onehot, min_support=min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=min_confidence)
        if not rules.empty:
            top_rules = rules.sort_values("confidence", ascending=False).head(10)
            st.write("**Top-10 Association Rules (by confidence):**")
            st.dataframe(top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
        else:
            st.info("No rules found with current support and confidence.")

elif tab == "Regression Insights":
    st.header("Regression: Linear, Ridge, Lasso, DT")
    df = load_data()
    num_targets = df.select_dtypes(include=np.number).columns.tolist()
    if not num_targets:
        st.warning("No numeric targets found.")
    else:
        target = st.selectbox("Select a numeric target variable for regression:", num_targets)
        X, y = preprocess_for_regression(df, target)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree Regressor": DecisionTreeRegressor()
        }
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, pred)
            results.append({"Model": name, "RMSE": rmse, "R2": r2})
        st.subheader("Regression Model Comparison")
        st.dataframe(pd.DataFrame(results).set_index("Model"))

        st.subheader("Predicted vs Actual (Scatter Plot)")
        selected_model = st.selectbox("Choose model for plot", list(models.keys()))
        pred = models[selected_model].predict(X_test)
        fig, ax = plt.subplots()
        ax.scatter(y_test, pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Predicted vs Actual for {selected_model}")
        st.pyplot(fig)
        st.success("These insights reflect model performance and prediction accuracy.")
