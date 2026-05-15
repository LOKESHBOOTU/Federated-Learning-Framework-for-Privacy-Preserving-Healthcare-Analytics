from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.errors import StreamlitAPIException

from src.artifacts import artifact_paths, fixed_config_for_artifact, get_or_create_fixed_cdc_results, make_client_summary
from src.config import DATASET_CANDIDATES, HEART_DISEASE_DATASET_CANDIDATES, RAW_DATA_DIR
from src.data import prepare_healthcare_data
from src.federated import FederatedConfig, run_federated_training


st.set_page_config(
    page_title="Privacy-Preserving Federated Healthcare Analytics",
    page_icon="",
    layout="wide",
)

st.markdown(
    """
    <style>
    .prediction-panel {
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 1.25rem 1.25rem 0.4rem 1.25rem;
        background: rgba(255, 255, 255, 0.025);
        margin-bottom: 1rem;
    }
    .prediction-result {
        border-radius: 6px;
        padding: 0.9rem 1rem;
        background: rgba(20, 110, 65, 0.26);
        border: 1px solid rgba(64, 180, 115, 0.35);
        font-weight: 600;
        margin-top: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def dataset_cache_key(dataset_path: str | None = None) -> tuple[str, float]:
    if dataset_path:
        path = Path(dataset_path)
        if path.exists():
            return str(path), path.stat().st_mtime
    for path, _ in DATASET_CANDIDATES:
        if path.exists():
            return str(path), path.stat().st_mtime
    csv_files = sorted(RAW_DATA_DIR.glob("*.csv"))
    if csv_files:
        path = csv_files[0]
        return str(path), path.stat().st_mtime
    return "missing", 0.0


def artifact_cache_key(artifact_key: str) -> tuple[str, float]:
    paths = artifact_paths(artifact_key)
    metadata = paths["metadata"]
    if metadata.exists():
        return str(metadata), metadata.stat().st_mtime
    return str(metadata), 0.0


@st.cache_data(show_spinner=False)
def load_data(dataset_key, target_name, dataset_label, dataset_path=None):
    path = None if not dataset_path else Path(dataset_path)
    return prepare_healthcare_data(path=path, target_name=target_name, dataset_label=dataset_label)


@st.cache_resource(show_spinner=False)
def load_saved_cdc_experiment(dataset_key, target_name, dataset_label, artifact_key, artifact_key_for_cache, dataset_path=None):
    data = load_data(dataset_key, target_name, dataset_label, dataset_path)
    history_df, client_summary, model, baseline_results, loaded_from_disk = get_or_create_fixed_cdc_results(
        data, artifact_key
    )
    return history_df, client_summary, model, baseline_results, loaded_from_disk


@st.cache_resource(show_spinner=False)
def run_simulation_experiment(dataset_key, target_name, dataset_label, dataset_path, clients, rounds, local_epochs, partition, use_dp, noise, lr):
    data = load_data(dataset_key, target_name, dataset_label, dataset_path)
    config = FederatedConfig(
        num_clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
        partition=partition,
        use_dp=use_dp,
        noise_multiplier=noise,
        learning_rate=lr,
    )
    model, history, client_parts = run_federated_training(
        data.X_train, data.y_train, data.X_test, data.y_test, config
    )
    return pd.DataFrame(history), make_client_summary(client_parts), model


def risk_band(risk: float) -> tuple[str, str, str]:
    if risk >= 0.7:
        return "High Risk", "Urgent clinical follow-up", "Prioritize HbA1c testing and physician review."
    if risk >= 0.4:
        return "Moderate Risk", "Lifestyle counseling", "Recommend diet, activity, and repeat screening."
    return "Low Risk", "Routine monitoring", "Continue periodic preventive screening."


PREDICTION_TASKS = {
    "Diabetes Prediction": {
        "artifact_key": "diabetes",
        "target_name": "Diabetes_binary",
        "dataset_label": "CDC Diabetes Health Indicators 50/50 Balanced",
        "title": "Diabetes Prediction",
        "button": "Diabetes Test Result",
        "positive_label": "Diabetes / Prediabetes",
        "negative_label": "No Diabetes",
        "result_label": "Predicted diabetes risk",
        "dataset_path_candidates": [],
        "main_features": [
            "Age",
            "BMI",
            "GenHlth",
            "HighBP",
            "HighChol",
            "DiffWalk",
            "HeartDiseaseorAttack",
            "PhysHlth",
            "PhysActivity",
            "Sex",
        ],
    },
    "Heart Disease Prediction": {
        "artifact_key": "heart_disease",
        "target_name": "condition",
        "dataset_label": "Heart Disease Cleveland UCI",
        "title": "Heart Disease Prediction",
        "button": "Heart Disease Test Result",
        "positive_label": "Heart disease present",
        "negative_label": "No heart disease",
        "result_label": "Predicted heart disease risk",
        "dataset_path_candidates": [str(path) for path, _ in HEART_DISEASE_DATASET_CANDIDATES],
        "main_features": [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "thalach",
            "exang",
        ],
    },
}


def resolve_task_dataset_path(task: dict) -> str | None:
    for candidate in task.get("dataset_path_candidates", []):
        path = Path(candidate)
        if path.exists():
            return str(path)
    return None


FEATURE_LABELS = {
    "Age": "Age category",
    "BMI": "BMI value",
    "GenHlth": "General health",
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "DiffWalk": "Difficulty walking",
    "HeartDiseaseorAttack": "Heart disease or attack",
    "PhysHlth": "For how many days was your Physical Health not good?",
    "PhysActivity": "Physical activity",
    "Stroke": "Stroke history",
    "Education": "Education category",
    "Income": "Income category",
    "Smoker": "Smoker",
    "Fruits": "Consumes fruit",
    "Veggies": "Consumes vegetables",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has healthcare access",
    "NoDocbcCost": "Could not see doctor due to cost",
    "MentHlth": "For how many days was your Mental Health not good?",
    "Sex": "Sex",
    "CholCheck": "Cholesterol check",
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG result",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression from exercise",
    "slope": "Peak exercise ST slope",
    "ca": "Major vessels colored by fluoroscopy",
    "thal": "Thalassemia result",
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1,
    "Very good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5,
}


EDUCATION_OPTIONS = {
    "Never attended school or only kindergarten": 1,
    "Grades 1-8": 2,
    "Grades 9-11": 3,
    "Grade 12 / GED / high school graduate": 4,
    "College 1-3 years": 5,
    "College 4 years or more": 6,
}


INCOME_OPTIONS = {
    "Less than $10,000": 1,
    "$10,000-$15,000": 2,
    "$15,000-$20,000": 3,
    "$20,000-$25,000": 4,
    "$25,000-$35,000": 5,
    "$35,000-$50,000": 6,
    "$50,000-$75,000": 7,
    "$75,000 or more": 8,
}


SEX_OPTIONS = {
    "Female": 0,
    "Male": 1,
}

HEART_FEATURE_OPTIONS = {
    "cp": {
        "Heart-related chest pain": 0,
        "Unusual chest pain": 1,
        "Chest pain not likely from heart": 2,
        "No chest pain": 3,
    },
    "restecg": {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2,
    },
    "slope": {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2,
    },
    "ca": {
        "0 vessels": 0,
        "1 vessel": 1,
        "2 vessels": 2,
        "3 vessels": 3,
    },
    "thal": {
        "Normal": 0,
        "Fixed defect": 1,
        "Reversible defect": 2,
    },
}


def age_to_cdc_category(age: int) -> int:
    if age <= 24:
        return 1
    if age <= 29:
        return 2
    if age <= 34:
        return 3
    if age <= 39:
        return 4
    if age <= 44:
        return 5
    if age <= 49:
        return 6
    if age <= 54:
        return 7
    if age <= 59:
        return 8
    if age <= 64:
        return 9
    if age <= 69:
        return 10
    if age <= 74:
        return 11
    if age <= 79:
        return 12
    return 13


def cdc_category_to_age(category: float) -> int:
    representative_ages = {
        1: 21,
        2: 27,
        3: 32,
        4: 37,
        5: 42,
        6: 47,
        7: 52,
        8: 57,
        9: 62,
        10: 67,
        11: 72,
        12: 77,
        13: 80,
    }
    return representative_ages.get(int(round(category)), 45)


def feature_input(feature: str, default: float, prefix: str):
    label = FEATURE_LABELS.get(feature, feature)
    key = f"{prefix}_{feature}"
    if feature in HEART_FEATURE_OPTIONS:
        options = HEART_FEATURE_OPTIONS[feature]
        labels = list(options.keys())
        values = list(options.values())
        default_value = int(round(default))
        default_index = values.index(default_value) if default_value in values else 0
        selected = st.selectbox(label, labels, index=default_index, key=key)
        return options[selected]
    if feature in {"fbs", "exang"}:
        return int(st.toggle(label, value=bool(round(default)), key=key))
    if feature == "age":
        return st.number_input(label, min_value=18, max_value=120, value=int(round(default)), step=1, format="%d", key=key)
    if feature in {"trestbps", "chol", "thalach"}:
        return st.number_input(label, min_value=0, value=int(round(default)), step=1, format="%d", key=key)
    if feature == "oldpeak":
        return st.number_input(label, min_value=0.0, value=float(default), step=0.1, format="%.1f", key=key)
    if feature in {"HighBP", "HighChol", "DiffWalk", "HeartDiseaseorAttack", "PhysActivity"}:
        return int(st.toggle(label, value=bool(round(default)), key=key))
    if feature in {"Smoker", "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "CholCheck"}:
        return int(st.toggle(label, value=bool(round(default)), key=key))
    if feature in {"Sex", "sex"}:
        option_labels = list(SEX_OPTIONS.keys())
        default_index = max(0, min(1, int(round(default))))
        selected = st.selectbox(label, option_labels, index=default_index, key=key)
        return SEX_OPTIONS[selected]
    if feature == "Age":
        try:
            real_age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=None,
                step=1,
                format="%d",
                key=key,
                placeholder="Enter age",
                help="The CDC dataset stores age as ranges, so this real age is converted to the matching CDC age group internally.",
            )
        except StreamlitAPIException:
            real_age = st.number_input(
                "Age",
                min_value=18,
                max_value=120,
                value=18,
                step=1,
                format="%d",
                key=key,
                help="The CDC dataset stores age as ranges, so this real age is converted to the matching CDC age group internally.",
            )
        if real_age is None:
            return None
        return age_to_cdc_category(int(real_age))
    if feature == "GenHlth":
        option_labels = list(GENERAL_HEALTH_OPTIONS.keys())
        default_index = max(0, min(4, int(round(default)) - 1))
        selected = st.selectbox("General health", option_labels, index=default_index, key=key)
        return GENERAL_HEALTH_OPTIONS[selected]
    if feature == "Education":
        option_labels = list(EDUCATION_OPTIONS.keys())
        default_index = max(0, min(5, int(round(default)) - 1))
        selected = st.selectbox(label, option_labels, index=default_index, key=key)
        return EDUCATION_OPTIONS[selected]
    if feature == "Income":
        option_labels = list(INCOME_OPTIONS.keys())
        default_index = max(0, min(7, int(round(default)) - 1))
        selected = st.selectbox(label, option_labels, index=default_index, key=key)
        return INCOME_OPTIONS[selected]
    if feature in {"PhysHlth", "MentHlth"}:
        return st.number_input(
            label,
            min_value=0,
            max_value=30,
            value=int(round(default)),
            step=1,
            format="%d",
            key=key,
        )
    return st.number_input(label, value=float(default), step=1.0, key=key)


def patient_input_form(data, prefix: str = "patient", main_features: list[str] | None = None) -> pd.DataFrame:
    values = {}
    medians = data.frame[data.feature_names].median(numeric_only=True)
    main_features = main_features or data.feature_names[: min(10, len(data.feature_names))]
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, feature in enumerate(main_features):
        if feature not in data.feature_names:
            continue
        with cols[idx % 3]:
            values[feature] = feature_input(feature, float(medians[feature]), prefix)

    advanced_features = [feature for feature in data.feature_names if feature not in main_features]
    with st.expander("Advanced optional details"):
        adv_cols = st.columns(3)
        for idx, feature in enumerate(advanced_features):
            with adv_cols[idx % 3]:
                values[feature] = feature_input(feature, float(medians[feature]), prefix)
    st.markdown("</div>", unsafe_allow_html=True)

    for feature in data.feature_names:
        values.setdefault(feature, float(medians[feature]))
    age_feature = "Age" if "Age" in data.feature_names else "age" if "age" in data.feature_names else None
    if age_feature and values.get(age_feature) is None:
        return None
    return pd.DataFrame([[values[feature] for feature in data.feature_names]], columns=data.feature_names)


def predict_patient(model, scaler, patient_df: pd.DataFrame) -> float:
    user_x = scaler.transform(patient_df.values)
    return float(model.predict_proba(user_x)[0])


DEFAULT_SIMULATION_SETTINGS = {
    "clients": 5,
    "rounds": 15,
    "local_epochs": 2,
    "partition": "iid",
    "use_dp": True,
    "noise": 0.05,
    "lr": 0.03,
}


def get_committed_simulation_settings() -> dict:
    if "simulation_settings" not in st.session_state:
        st.session_state.simulation_settings = DEFAULT_SIMULATION_SETTINGS.copy()
    return st.session_state.simulation_settings


with st.sidebar:
    st.header("Multiple Disease Prediction System")
    selected_task_name = st.radio("Prediction module", list(PREDICTION_TASKS.keys()))
    selected_task = PREDICTION_TASKS[selected_task_name]
    st.divider()
    st.header("Experiment Controls")
    run_mode = st.radio(
        "Model source",
        ["Saved trained results", "Simulate hospitals"],
        help="Saved trained results reuse the fixed trained model. Simulate hospitals retrains for the selected settings.",
    )

    if run_mode == "Simulate hospitals":
        applied_settings = get_committed_simulation_settings()
        st.subheader("Hospital Simulation")
        pending_clients = st.slider("Simulated hospitals", 2, 10, applied_settings["clients"])
        pending_rounds = st.slider("Federated rounds", 3, 50, applied_settings["rounds"])
        pending_local_epochs = st.slider("Local epochs per hospital", 1, 5, applied_settings["local_epochs"])
        pending_partition = st.radio(
            "Data distribution",
            ["iid", "non-iid"],
            index=["iid", "non-iid"].index(applied_settings["partition"]),
            horizontal=True,
        )
        pending_use_dp = st.toggle("Differential privacy", value=applied_settings["use_dp"])
        pending_noise = st.slider(
            "DP noise multiplier",
            0.0,
            0.5,
            applied_settings["noise"],
            0.01,
            disabled=not pending_use_dp,
        )
        pending_lr = st.slider("Learning rate", 0.005, 0.1, applied_settings["lr"], 0.005)
        pending_settings = {
            "clients": pending_clients,
            "rounds": pending_rounds,
            "local_epochs": pending_local_epochs,
            "partition": pending_partition,
            "use_dp": pending_use_dp,
            "noise": pending_noise,
            "lr": pending_lr,
        }

        if st.button("Train simulation", type="primary", use_container_width=True):
            st.session_state.simulation_settings = pending_settings
            applied_settings = pending_settings

        if pending_settings != applied_settings:
            st.warning("Settings changed. Click Train simulation to retrain with these values.")
        else:
            st.success("Current settings are applied.")

        clients = applied_settings["clients"]
        rounds = applied_settings["rounds"]
        local_epochs = applied_settings["local_epochs"]
        partition = applied_settings["partition"]
        use_dp = applied_settings["use_dp"]
        noise = applied_settings["noise"]
        lr = applied_settings["lr"]
        st.caption(
            f"Applied: {clients} hospitals, {rounds} rounds, {local_epochs} local epochs, "
            f"{partition.upper()}, DP {'on' if use_dp else 'off'}."
        )
    else:
        fixed_config = fixed_config_for_artifact(selected_task["artifact_key"])
        clients = fixed_config.num_clients
        rounds = fixed_config.rounds
        local_epochs = fixed_config.local_epochs
        partition = fixed_config.partition
        use_dp = fixed_config.use_dp
        noise = fixed_config.noise_multiplier
        lr = fixed_config.learning_rate
        st.info(
            f"Using fixed saved settings: {clients} hospitals, {rounds} federated rounds, "
            f"{local_epochs} local epochs, {partition.upper()}, DP {'enabled' if use_dp else 'disabled'}."
        )

try:
    selected_dataset_path = resolve_task_dataset_path(selected_task)
    current_dataset_key = dataset_cache_key(selected_dataset_path)
    data = load_data(
        current_dataset_key,
        selected_task["target_name"],
        selected_task["dataset_label"],
        selected_dataset_path,
    )
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

st.title("Federated Learning Framework for Privacy-Preserving Healthcare Analytics")
st.caption(f"Current module: {selected_task['title']} with local hospital training and central FedAvg aggregation.")

if run_mode == "Saved trained results":
    with st.spinner("Loading saved trained results. If missing, training once and saving them..."):
        history_df, client_summary, fed_model, baseline_results, loaded_from_disk = load_saved_cdc_experiment(
            current_dataset_key,
            selected_task["target_name"],
            selected_task["dataset_label"],
            selected_task["artifact_key"],
            artifact_cache_key(selected_task["artifact_key"]),
            selected_dataset_path,
        )
    model_source = "Saved trained results" if loaded_from_disk else "Newly trained results saved for reuse"
    prediction_source = "the saved fixed federated model"
else:
    with st.spinner("Training hospital simulation for the selected settings..."):
        history_df, client_summary, fed_model = run_simulation_experiment(
            current_dataset_key,
            selected_task["target_name"],
            selected_task["dataset_label"],
            selected_dataset_path,
            clients,
            rounds,
            local_epochs,
            partition,
            use_dp,
            noise,
            lr,
        )
    baseline_results = None
    model_source = "Live hospital simulation"
    prediction_source = "the current hospital simulation model"

tabs = st.tabs(
    [
        "Prediction Demo",
        "Model Comparison",
        "Dataset",
        "Federated Training",
        "Presenter Guide",
    ]
)

with tabs[0]:
    st.header(selected_task["title"])
    with st.form("single_patient_prediction"):
        user_df = patient_input_form(data, prefix="single", main_features=selected_task["main_features"])
        submitted = st.form_submit_button(selected_task["button"])

    if submitted:
        if user_df is None:
            st.error(f"Please enter the patient's age before generating the {selected_task['title'].lower()} result.")
        else:
            risk = predict_patient(fed_model, data.scaler, user_df)
            band, action, note = risk_band(risk)
            st.markdown(
                f'<div class="prediction-result">{selected_task["result_label"]}: {risk:.1%} | {band} | {action}</div>',
                unsafe_allow_html=True,
            )
            st.progress(min(max(risk, 0.0), 1.0))
            st.info(note)
    else:
        st.caption(f"Enter the patient details and click {selected_task['button']} to generate the prediction.")

with tabs[1]:
    rows = [{"Model": name, **metrics} for name, metrics in (baseline_results or {}).items()]
    rows.append({"Model": "Federated MLP + FedAvg" + (" + DP" if use_dp else ""), **history_df.iloc[-1].to_dict()})
    comparison = pd.DataFrame(rows)
    display_cols = ["Model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    st.dataframe(comparison[display_cols], use_container_width=True)
    st.plotly_chart(
        px.bar(comparison, x="Model", y=["accuracy", "precision", "recall", "f1", "roc_auc"], barmode="group"),
        use_container_width=True,
    )

with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset", data.name)
    c2.metric("Rows", f"{len(data.frame):,}")
    c3.metric("Features", len(data.feature_names))
    c4.metric("Positive class", f"{data.y_train.mean():.1%}")
    st.caption(f"Model source: {model_source}. Predictions are made using {prediction_source}.")

    st.subheader("Dataset Preview")
    st.dataframe(data.frame.head(20), use_container_width=True)

    st.subheader("Class Balance")
    counts = data.frame[data.target_name].value_counts().sort_index()
    class_balance = pd.DataFrame(
        {
            "Class": [selected_task["negative_label"], selected_task["positive_label"]],
            "Records": [int(counts.get(0, 0)), int(counts.get(1, 0))],
        }
    )
    class_balance["Percentage"] = class_balance["Records"] / class_balance["Records"].sum()

    b1, b2, b3 = st.columns(3)
    b1.metric("Total records", f"{class_balance['Records'].sum():,}")
    b2.metric(selected_task["negative_label"], f"{class_balance.loc[0, 'Records']:,}", f"{class_balance.loc[0, 'Percentage']:.1%}")
    b3.metric(
        selected_task["positive_label"],
        f"{class_balance.loc[1, 'Records']:,}",
        f"{class_balance.loc[1, 'Percentage']:.1%}",
    )

    fig = px.bar(
        class_balance,
        x="Records",
        y="Class",
        color="Class",
        orientation="h",
        text=class_balance.apply(lambda row: f"{row['Records']:,} ({row['Percentage']:.1%})", axis=1),
        color_discrete_map={
            selected_task["negative_label"]: "#69b3f2",
            selected_task["positive_label"]: "#ff6b6b",
        },
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Number of patient records",
        yaxis_title="",
        bargap=0.35,
        height=330,
        margin=dict(l=10, r=20, t=20, b=20),
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "This chart shows how many records are in each target class for the selected prediction module. "
        "If one class is much larger than the other, F1 score, recall, and ROC-AUC are more useful than accuracy alone."
    )

    st.info(
        f"The dashboard is using `{data.source_path}`. Saved results train once with fixed settings, "
        "then reload the saved artifacts on later runs."
    )

with tabs[3]:
    last = history_df.iloc[-1]
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{last['accuracy']:.3f}")
    m2.metric("Precision", f"{last['precision']:.3f}")
    m3.metric("Recall", f"{last['recall']:.3f}")
    m4.metric("F1 Score", f"{last['f1']:.3f}")
    m5.metric("ROC-AUC", f"{last['roc_auc']:.3f}")

    st.subheader("Training Progress")
    chart_df = history_df.melt(
        id_vars=["round"],
        value_vars=["accuracy", "f1", "roc_auc", "loss"],
        var_name="Metric",
        value_name="Value",
    )
    st.plotly_chart(px.line(chart_df, x="round", y="Value", color="Metric", markers=True), use_container_width=True)

    st.subheader("Simulated Hospital Distribution")
    st.caption(
        f"Current setup: {clients} hospitals, {rounds} federated rounds, "
        f"{local_epochs} local epochs per hospital, {partition.upper()} partition."
    )
    st.dataframe(client_summary, use_container_width=True)
    st.plotly_chart(
        px.bar(client_summary, x="Client", y="Records", color="Positive Rate", text_auto=True),
        use_container_width=True,
    )

with tabs[4]:
    st.subheader("Presentation Flow")
    st.markdown(
        """
        **Problem:** Hospitals cannot freely share patient records because healthcare data is sensitive.

        **Solution:** Federated learning trains a shared model without moving raw patient data.

        **Model:** A small MLP is trained locally at every hospital and aggregated through FedAvg.

        **Privacy Layer:** Differential privacy clips large updates and adds noise before the server sees them.

        **Evaluation:** Compare centralized baselines, federated learning, IID data, non-IID data, and DP/no-DP settings.

        **Major Project Angle:** The framework supports two prediction modules: Diabetes Risk and Heart Disease,
        each trained on the CDC health indicators dataset with dedicated targets and feature pipelines.
        """
    )

    st.subheader("Recommended Demo Steps")
    st.markdown(
        """
        1. Start on the Dataset tab and explain why the selected disease prediction is clinically relevant.
        2. Switch between Diabetes and Heart Disease modules in the sidebar to compare their results.
        3. Open Federated Training and show hospital-wise distribution.
        4. Toggle Differential Privacy and explain the privacy-accuracy tradeoff.
        5. Switch IID to non-IID to show real-world hospital heterogeneity.
        6. Use Model Comparison to justify the final model.
        7. End with Prediction Demo to make the project feel practical.
        """
    )
