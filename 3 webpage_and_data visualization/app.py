import json
import gradio as gr
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import joblib
from llm import chat
import webbrowser
import threading

# load the CSS and JS files
CSS = open("./static/app.css", 'r', encoding='utf-8').read()
JAVASCRIPT = open("./static/app.js", 'r', encoding='utf-8').read()
THEME = gr.themes.Default(
    font=("ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=("Cascadia Code", "Consolas", "ui-monospace", "Consolas", "monospace")
)

CACHED_DF = None

def get_cdc_data():
    global CACHED_DF
    df = pd.read_csv('archive/2022/heart_2022_no_nans.csv') if CACHED_DF is None else CACHED_DF
    if CACHED_DF is None:
        CACHED_DF = df
    return CACHED_DF.copy()

get_cdc_data()

CACHED_CLASSIFIER = None

def get_model():
    global CACHED_CLASSIFIER
    rf = joblib.load('static/heart_attack_model.pkl') if CACHED_CLASSIFIER is None else CACHED_CLASSIFIER
    if CACHED_CLASSIFIER is None:
        CACHED_CLASSIFIER = rf
    return CACHED_CLASSIFIER

get_model()

X = pd.get_dummies(get_cdc_data().drop('HadHeartAttack', axis=1), drop_first=True)

def query_cdc_data():
    df = get_cdc_data()
    return df[0:100] # first page
    
def generate_distribution_plots():
    df = get_cdc_data()
    fig_state = px.histogram(df, x='State', template='plotly_dark')
    fig_sex = px.histogram(df, x='Sex', template='plotly_dark')
    fig_health = px.histogram(df, x='GeneralHealth', template='plotly_dark')
    fig_heart_attack = px.histogram(df, x='HadHeartAttack', template='plotly_dark')
    return fig_state, fig_sex, fig_health, fig_heart_attack

def get_importance():
    # analyze the coefficients of each factor, get the most relevant factors
    importance = get_model().feature_importances_
    features = X.columns

    # create a dataframe
    df_importance = pd.DataFrame({'features': features, 'importance': importance})

    # sort the dataframe
    df_importance = df_importance.sort_values('importance', ascending=False)

    return df_importance

# Function to generate the feature importances plot
def generate_feature_importances_plots():
    rf = get_model()

    # analyze the coefficients of each factor, get the most relevant factors
    importance = rf.feature_importances_
    features = X.columns

    # create a dataframe
    df_importance = pd.DataFrame({'features': features, 'importance': importance})

    # sort the dataframe
    df_importance = df_importance.sort_values('importance', ascending=False)

    fig = px.bar(
        df_importance.head(20).sort_values(by='importance', ascending=True),
        x='importance',
        y='features',
        orientation='h',
        color='importance',
        color_continuous_scale='RdYlGn',  # Red to Green color scale
        range_color=[-1, 1],              # Define the range for color scaling
        text='importance',                # Display importance values as text
        template='plotly_dark'            # Dark theme
    )
    fig.update_layout(
        height=1000,
        autosize=True,
        xaxis_title='Importance',
        yaxis_title='Features',
        coloraxis_colorbar=dict(
            title="Importance",
            tickvals=[-1, 0, 1],
            ticktext=["-1", "0", "1"]
        )
    )

    # Customize text for better readability
    fig.update_traces(
        texttemplate='%{text:.2f}',  # Format the text to two decimal places
        textposition='inside'       # Place the text outside the bars
    )
    return fig

def generate_risk_distribution_plots():
    df, rf = get_cdc_data(), get_model()
    y = df['HadHeartAttack']
    df['HadHeartAttack'] = y
    df['Risk'] = rf.predict_proba(X)[:, 1]
    fig = px.histogram(df, x='Risk', color='HadHeartAttack', marginal='box', title='Risk Distribution', template='plotly_dark')
    fig.update_layout(height=1000)
    return fig

def generate_plots():
    plot_state, plot_sex, plot_health, plot_heart_attack = generate_distribution_plots()
    plot_top_features = generate_feature_importances_plots()
    # plot_risk_distribution = generate_risk_distribution_plots()
    return plot_state, plot_sex, plot_health, plot_heart_attack, plot_top_features

# Function to get data for a specific page
def get_page_data(page_number = 1):
    df = get_cdc_data()
    page_size = 100
    start = (int(page_number) - 1) * page_size
    end = start + page_size
    return df.iloc[start:end]

def refresh_data():
    plot_state, plot_sex, plot_health, plot_heart_attack, plot_top_features = generate_plots()
    df = get_page_data()
    return plot_state, plot_sex, plot_health, plot_heart_attack, plot_top_features, df

def calcualte_risk(state, sex, general_health, physical_health_days, mental_health_days, last_checkup_time, physical_activities, sleep_hours, removed_teeth, had_angina, had_stroke, had_asthma, had_skin_cancer, had_copd, had_depressive_disorder, had_kidney_disease, had_arthritis, had_diabetes, deaf_or_hard_of_hearing, blind_or_vision_difficulty, difficulty_concentrating, difficulty_walking, difficulty_dressing_bathing, difficulty_errands, smoker_status, e_cigarette_usage, chest_scan, race_ethnicity_category, age_category, height_in_meters, weight_in_kilograms, bmi, alcohol_drinkers, hiv_testing, flu_vax_last_12, pneumo_vax_ever, tetanus_last_10_tdap, high_risk_last_year, covid_pos):
    rf = get_model()
    new_x = pd.DataFrame({
        'State': [state],
        'Sex': [sex],
        'GeneralHealth': [general_health],
        'PhysicalHealthDays': [physical_health_days],
        'MentalHealthDays': [mental_health_days],
        'LastCheckupTime': [last_checkup_time],
        'PhysicalActivities': [physical_activities],
        'SleepHours': [sleep_hours],
        'RemovedTeeth': [removed_teeth],
        'HadAngina': [had_angina],
        'HadStroke': [had_stroke],
        'HadAsthma': [had_asthma],
        'HadSkinCancer': [had_skin_cancer],
        'HadCOPD': [had_copd],
        'HadDepressiveDisorder': [had_depressive_disorder],
        'HadKidneyDisease': [had_kidney_disease],
        'HadArthritis': [had_arthritis],
        'HadDiabetes': [had_diabetes],
        'DeafOrHardOfHearing': [deaf_or_hard_of_hearing],
        'BlindOrVisionDifficulty': [blind_or_vision_difficulty],
        'DifficultyConcentrating': [difficulty_concentrating],
        'DifficultyWalking': [difficulty_walking],
        'DifficultyDressingBathing': [difficulty_dressing_bathing],
        'DifficultyErrands': [difficulty_errands],
        'SmokerStatus': [smoker_status],
        'ECigaretteUsage': [e_cigarette_usage],
        'ChestScan': [chest_scan],
        'RaceEthnicityCategory': [race_ethnicity_category],
        'AgeCategory': [age_category],
        'HeightInMeters': [height_in_meters],
        'WeightInKilograms': [weight_in_kilograms],
        'BMI': [bmi],
        'AlcoholDrinkers': [alcohol_drinkers],
        'HIVTesting': [hiv_testing],
        'FluVaxLast12': [flu_vax_last_12],
        'PneumoVaxEver': [pneumo_vax_ever],
        'TetanusLast10Tdap': [tetanus_last_10_tdap],
        'HighRiskLastYear': [high_risk_last_year],
        'CovidPos': [covid_pos]
    })
    merged_x = get_cdc_data().drop('HadHeartAttack', axis=1)
    merged_x = pd.concat([merged_x, new_x], ignore_index=True)
    X = pd.get_dummies(merged_x, drop_first=True)
    risk = rf.predict_proba(X.iloc[-1:])[:, 1].tolist()[0]
    all_factors = X.iloc[-1:].to_dict(orient='records')[0]
    try:
        summary = get_insight_markdown(risk, checked_summary(all_factors))
    except:
        summary = "Failed to generate health summary. Please read the factors manually from other tabs."
    return risk, summary

def checked_summary(all_factors):
    importance = get_importance()
    # list top 20 importance factors
    top_20 = importance.head(20).to_dict(orient='records')
    for factor in top_20:
        factor_name = factor['features']
        if all_factors.get(factor_name) == False:
            factor['user'] = {
                'hit': False
            }
        else:
            if all_factors.get(factor_name) == True:
                factor['user'] = {
                    'hit': True
                }
            else:
                factor['user'] = {
                    'hit': None,
                    'value': all_factors.get(factor_name)
                }
    return top_20

def get_insight_markdown(percentage, checklist):
    response = chat(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": open("prompt.md", "r", encoding="utf-8").read().strip(),
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f'General Risk of Heart Disease: {percentage}\n' + json.dumps(checklist, indent=4, ensure_ascii=False),
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def load_random_data(should_pick_disease = None):
    should_pick_disease = np.random.choice([True, False], p=[0.5, 0.5]) if should_pick_disease is None else should_pick_disease
    df = get_cdc_data()
    random_data = df[df['HadHeartAttack'] == 'Yes'].sample(1) if should_pick_disease else df[df['HadHeartAttack'] == 'No'].sample(1)
    random_data = random_data.iloc[0].to_dict()
    state = random_data['State']
    sex = random_data['Sex']
    general_health = random_data['GeneralHealth']
    physical_health_days = random_data['PhysicalHealthDays']
    mental_health_days = random_data['MentalHealthDays']
    last_checkup_time = random_data['LastCheckupTime']
    physical_activities = random_data['PhysicalActivities']
    sleep_hours = random_data['SleepHours']
    removed_teeth = random_data['RemovedTeeth']
    had_angina = random_data['HadAngina']
    had_stroke = random_data['HadStroke']
    had_asthma = random_data['HadAsthma']
    had_skin_cancer = random_data['HadSkinCancer']
    had_copd = random_data['HadCOPD']
    had_depressive_disorder = random_data['HadDepressiveDisorder']
    had_kidney_disease = random_data['HadKidneyDisease']
    had_arthritis = random_data['HadArthritis']
    had_diabetes = random_data['HadDiabetes']
    deaf_or_hard_of_hearing = random_data['DeafOrHardOfHearing']
    blind_or_vision_difficulty = random_data['BlindOrVisionDifficulty']
    difficulty_concentrating = random_data['DifficultyConcentrating']
    difficulty_walking = random_data['DifficultyWalking']
    difficulty_dressing_bathing = random_data['DifficultyDressingBathing']
    difficulty_errands = random_data['DifficultyErrands']
    smoker_status = random_data['SmokerStatus']
    e_cigarette_usage = random_data['ECigaretteUsage']
    chest_scan = random_data['ChestScan']
    race_ethnicity_category = random_data['RaceEthnicityCategory']
    age_category = random_data['AgeCategory']
    height_in_meters = random_data['HeightInMeters']
    weight_in_kilograms = random_data['WeightInKilograms']
    bmi = random_data['BMI']
    alcohol_drinkers = random_data['AlcoholDrinkers']
    hiv_testing = random_data['HIVTesting']
    flu_vax_last_12 = random_data['FluVaxLast12']
    pneumo_vax_ever = random_data['PneumoVaxEver']
    tetanus_last_10_tdap = random_data['TetanusLast10Tdap']
    high_risk_last_year = random_data['HighRiskLastYear']
    covid_pos = random_data['CovidPos']
    return state, sex, general_health, physical_health_days, mental_health_days, last_checkup_time, physical_activities, sleep_hours, removed_teeth, had_angina, had_stroke, had_asthma, had_skin_cancer, had_copd, had_depressive_disorder, had_kidney_disease, had_arthritis, had_diabetes, deaf_or_hard_of_hearing, blind_or_vision_difficulty, difficulty_concentrating, difficulty_walking, difficulty_dressing_bathing, difficulty_errands, smoker_status, e_cigarette_usage, chest_scan, race_ethnicity_category, age_category, height_in_meters, weight_in_kilograms, bmi, alcohol_drinkers, hiv_testing, flu_vax_last_12, pneumo_vax_ever, tetanus_last_10_tdap, high_risk_last_year, covid_pos

def calcualte_risk_display(*args):
    risk, summary = calcualte_risk(*args)
    # return red if risk is high, yellow if medium 0.4, green if low
    color = 'red' if risk > 0.6 else 'yellow' if risk > 0.4 else 'green'
    # return %
    return f"Risk: <font style=\"color: {color} !important;\">{risk * 100:.2f}%</font>", summary

# create the Gradio block
with gr.Blocks(title="Heart Disease Forcast & Risk Measuring Engine", theme=THEME, css=CSS, js=JAVASCRIPT) as block:
    gr.Markdown("""<center><b><font size=8 class="shadow-text">Heart Disease Forcast & Risk Measuring Engine</center>""")
    gr.Markdown("""<center><font size=4 class="shadow-text">Based on Random Forest models, version: 2024-12-10</center>""")
    display_button = gr.Button("Refresh Data", variant='primary', elem_classes=["fit-content", "centered"], elem_id="refresh")
    with gr.Tab("💻 Distribution"):
        gr.Markdown("Over all distribution of the data is shown below. Gender is near equal, with a slight majority over **female**. States are rather interesting, with **Washington** being the most represented state, and the least represented state being **Virgin Islands**. The general health of the population is rather good, with a slight majority of people having **very good** health. The distribution of heart attacks is **not balanced**, with a majority of people not having had a heart attack.")
        with gr.Row():
            with gr.Column():
                plot_state = gr.Plot(label="Distribution of States")
            with gr.Column():
                plot_sex = gr.Plot(label="Distribution of Sex")
        with gr.Row():
            with gr.Column():
                plot_health = gr.Plot(label="Distribution of General Health")
            with gr.Column():
                plot_heart_attack = gr.Plot(label="Distribution of Heart Attacks")

    # gr.Markdown("# Top 20 Feature Importances")
    with gr.Tab("📊 Feature Importances"):
        gr.Markdown("The top 20 feature importances are shown below. The most important feature is **Had Angina** and **Chest Scan**, it makes sense that people having angina is a strong indicator of heart disease, and people who felt worry about their heart are more likely to have chest scans. No significant correlation observed between **Drinking**, **Smoking** and **Had Heart Attack**, differs to the common belief.")
        with gr.Row():
            plot_top_features = gr.Plot(label="Top 20 Feature Importances")
    with gr.Tab("🔎 Tracking Data"):
        with gr.Row(equal_height=True):
            left_button = gr.Button("◀️Previous")
            page_number = gr.Number(value=1, label="Page Number", interactive=True)
            right_button = gr.Button("Next ▶️") # ▶️
            display_data_button = gr.Button("🔄 Display", variant='primary', elem_classes=["fit-content"])
        datatable = gr.Dataframe(elem_classes=["data-table"], label="CDC Data - 2022 No NaNs", row_count=100)

        display_data_button.click(get_page_data, outputs=datatable)

        # Update datatable when page number changes
        page_number.change(get_page_data, inputs=page_number, outputs=datatable)

        # Navigate to previous page
        def prev_page(page_number):
            new_page = max(int(page_number) - 1, 1)
            if new_page < 1:
                new_page = 1
            return new_page

        left_button.click(prev_page, inputs=page_number, outputs=[page_number])

        # Navigate to next page
        def next_page(page_number):
            new_page = int(page_number) + 1
            return new_page

        right_button.click(next_page, inputs=page_number, outputs=[page_number])
    with gr.Tab("🏃‍♂️ Risk Distribution"):
        gr.Markdown("The risk distribution is shown below. The risk is calculated using the Random Forest model, and is calculated as the probability of having a heart attack. We can see clearly that people with risk over 40 percent are more likely to have a heart attack, with 80 percent being the median. People with 0 percent risk are less likely to have a heart attack.")
        gr.Markdown("The calculation of the risk is based on the Random Forest model, thus tooks a lot of factors into account. Please click the refresh button to update the risk distribution manually.")
        plot_risk_distribution_button = gr.Button("🔄 Refresh (about 30 seconds)", elem_classes=["fit-content"])
        plot_risk_distribution = gr.Plot(label="Risk Distribution")
        plot_risk_distribution_button.click(generate_risk_distribution_plots, outputs=plot_risk_distribution)
    with gr.Tab("🤖 Risk Factoring"):
        gr.Markdown("This panel displays for a certain user, what the risk of them having a heart attack is. The model takes into account a lot of factors, so we provides a mean of randomly loading one from our database.")
        with gr.Row():
            load_data_button = gr.Button("Load Random Data", elem_classes=["fit-content"])
            load_healthy_data_button = gr.Button("Load Healthy Data", elem_classes=["fit-content"])
            load_unhealthy_data_button = gr.Button("Load Unhealthy Data", elem_classes=["fit-content"])
            risk_button = gr.Button("Calculate Risk", variant='primary', elem_classes=["fit-content"])
        with gr.Row():
            # 3:1 columns layout
            with gr.Column(scale=8, variant='panel'):
                with gr.Row():
                    with gr.Column(scale=4):
                        state = gr.Dropdown(interactive=True, label="State", choices=get_cdc_data()['State'].unique().tolist())
                        sex = gr.Dropdown(interactive=True, label="Sex", choices=get_cdc_data()['Sex'].unique().tolist())
                        general_health = gr.Dropdown(interactive=True, label="General Health", choices=get_cdc_data()['GeneralHealth'].unique().tolist())
                        physical_health_days = gr.Number(label="Physical Health Days")
                        mental_health_days = gr.Number(label="Mental Health Days")
                        last_checkup_time = gr.Dropdown(interactive=True, label="Last Checkup Time", choices=get_cdc_data()['LastCheckupTime'].unique().tolist())
                        physical_activities = gr.Dropdown(interactive=True, label="Physical Activities", choices=get_cdc_data()['PhysicalActivities'].unique().tolist())
                        sleep_hours = gr.Number(label="Sleep Hours")
                        removed_teeth = gr.Dropdown(interactive=True, label="Removed Teeth", choices=get_cdc_data()['RemovedTeeth'].unique().tolist())
                        had_angina = gr.Dropdown(interactive=True, label="Had Angina", choices=get_cdc_data()['HadAngina'].unique().tolist())
                    with gr.Column(scale=4):
                        had_stroke = gr.Dropdown(interactive=True, label="Had Stroke", choices=get_cdc_data()['HadStroke'].unique().tolist())
                        had_asthma = gr.Dropdown(interactive=True, label="Had Asthma", choices=get_cdc_data()['HadAsthma'].unique().tolist())
                        had_skin_cancer = gr.Dropdown(interactive=True, label="Had Skin Cancer", choices=get_cdc_data()['HadSkinCancer'].unique().tolist())
                        had_copd = gr.Dropdown(interactive=True, label="Had COPD", choices=get_cdc_data()['HadCOPD'].unique().tolist())
                        had_depressive_disorder = gr.Dropdown(interactive=True, label="Had Depressive Disorder", choices=get_cdc_data()['HadDepressiveDisorder'].unique().tolist())
                        had_kidney_disease = gr.Dropdown(interactive=True, label="Had Kidney Disease", choices=get_cdc_data()['HadKidneyDisease'].unique().tolist())
                        had_arthritis = gr.Dropdown(interactive=True, label="Had Arthritis", choices=get_cdc_data()['HadArthritis'].unique().tolist())
                        had_diabetes = gr.Dropdown(interactive=True, label="Had Diabetes", choices=get_cdc_data()['HadDiabetes'].unique().tolist())
                        deaf_or_hard_of_hearing = gr.Dropdown(interactive=True, label="Deaf Or Hard Of Hearing", choices=get_cdc_data()['DeafOrHardOfHearing'].unique().tolist())
                        blind_or_vision_difficulty = gr.Dropdown(interactive=True, label="Blind Or Vision Difficulty", choices=get_cdc_data()['BlindOrVisionDifficulty'].unique().tolist())
                    with gr.Column(scale=4):
                        difficulty_concentrating = gr.Dropdown(interactive=True, label="Difficulty Concentrating", choices=get_cdc_data()['DifficultyConcentrating'].unique().tolist())
                        difficulty_walking = gr.Dropdown(interactive=True, label="Difficulty Walking", choices=get_cdc_data()['DifficultyWalking'].unique().tolist())
                        difficulty_dressing_bathing = gr.Dropdown(interactive=True, label="Difficulty Dressing Bathing", choices=get_cdc_data()['DifficultyDressingBathing'].unique().tolist())
                        difficulty_errands = gr.Dropdown(interactive=True, label="Difficulty Errands", choices=get_cdc_data()['DifficultyErrands'].unique().tolist())
                        smoker_status = gr.Dropdown(interactive=True, label="Smoker Status", choices=get_cdc_data()['SmokerStatus'].unique().tolist())
                        e_cigarette_usage = gr.Dropdown(interactive=True, label="E-Cigarette Usage", choices=get_cdc_data()['ECigaretteUsage'].unique().tolist())
                        chest_scan = gr.Dropdown(interactive=True, label="Chest Scan", choices=get_cdc_data()['ChestScan'].unique().tolist())
                        race_ethnicity_category = gr.Dropdown(interactive=True, label="Race Ethnicity Category", choices=get_cdc_data()['RaceEthnicityCategory'].unique().tolist())
                        age_category = gr.Dropdown(interactive=True, label="Age Category", choices=get_cdc_data()['AgeCategory'].unique().tolist())
                    with gr.Column(scale=4):
                        height_in_meters = gr.Number(label="Height In Meters")
                        weight_in_kilograms = gr.Number(label="Weight In Kilograms")
                        bmi = gr.Number(label="BMI")
                        alcohol_drinkers = gr.Dropdown(interactive=True, label="Alcohol Drinkers", choices=get_cdc_data()['AlcoholDrinkers'].unique().tolist())
                        hiv_testing = gr.Dropdown(interactive=True, label="HIV Testing", choices=get_cdc_data()['HIVTesting'].unique().tolist())
                        flu_vax_last_12 = gr.Dropdown(interactive=True, label="Flu Vax in Last 12 Months", choices=get_cdc_data()['FluVaxLast12'].unique().tolist())
                        pneumo_vax_ever = gr.Dropdown(interactive=True, label="Pneumo Vax Ever", choices=get_cdc_data()['PneumoVaxEver'].unique().tolist())
                        tetanus_last_10_tdap = gr.Dropdown(interactive=True, label="Tetanus Last 10 Years", choices=get_cdc_data()['TetanusLast10Tdap'].unique().tolist())
                        high_risk_last_year = gr.Dropdown(interactive=True, label="High Risk Last Year", choices=get_cdc_data()['HighRiskLastYear'].unique().tolist())
                        covid_pos = gr.Dropdown(interactive=True, label="Covid Positive", choices=get_cdc_data()['CovidPos'].unique().tolist())
            
            all_factors = [state, sex, general_health, physical_health_days, mental_health_days, last_checkup_time, physical_activities, sleep_hours, removed_teeth, had_angina, had_stroke, had_asthma, had_skin_cancer, had_copd, had_depressive_disorder, had_kidney_disease, had_arthritis, had_diabetes, deaf_or_hard_of_hearing, blind_or_vision_difficulty, difficulty_concentrating, difficulty_walking, difficulty_dressing_bathing, difficulty_errands, smoker_status, e_cigarette_usage, chest_scan, race_ethnicity_category, age_category, height_in_meters, weight_in_kilograms, bmi, alcohol_drinkers, hiv_testing, flu_vax_last_12, pneumo_vax_ever, tetanus_last_10_tdap, high_risk_last_year, covid_pos]

            with gr.Column(scale=6):
                risk = gr.HTML("Risk: N/A", elem_classes=["risk-display"])
                summary = gr.Markdown(label="Insight & Summary")

            risk_button.click(calcualte_risk_display, inputs=all_factors, outputs=[risk, summary])
            load_data_button.click(load_random_data, outputs=all_factors)
            load_healthy_data_button.click(lambda: load_random_data(False), outputs=all_factors)
            load_unhealthy_data_button.click(lambda: load_random_data(True), outputs=all_factors)

    with gr.Tab("⚙️Settings"):
        gr.Markdown("# Run the model locally")
        gr.Code("import joblib\n\n# read the pkl\nmodel = joblib.load('static/heart_attack_model.pkl')\n\n# make predictions\ny_pred = model.predict(X_test)", language='python', label="Code")
        gr.Button("Download Model", link='/static/heart_attack_model.pkl', variant='primary')
    
    # Update plots when button is clicked
    display_button.click(
        fn=refresh_data,
        outputs=[plot_state, plot_sex, plot_health, plot_heart_attack, plot_top_features, datatable]
    )
    

# create the FastAPI app
app = FastAPI()

# create a static directory to store the static files
static_dir = Path('./static')
static_dir.mkdir(parents=True, exist_ok=True)

# mount FastAPI StaticFiles server
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# mount the gradio app
app = gr.mount_gradio_app(app, block, path="/")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:4242/")

# serve the app
if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=4242)
