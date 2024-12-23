# Webpage setup
@[youtube](https://www.youtube.com/watch?v=i-iC6KQ0HJg)

# Detailed Steps

Below are the detailed steps to run the application based on the `app.py` file:

## 1. Environment Setup
    ```
    # Install Python:
    # Ensure Python 3.7 or later is installed. Check the version using:
    python --version

    # Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate  # For Windows: venv\Scripts\activate

    # Install dependencies:
    # Create a `requirements.txt` file with the following content:
    # fastapi
    # uvicorn
    # gradio
    # pandas
    # numpy
    # matplotlib
    # plotly
    # joblib
    # Then, run:
    pip install -r requirements.txt
    ```

## 2. File Preparation
    ```
    # Verify project files:
    # Ensure the project directory contains the following files and folders:
    # - app.py: Main application file.
    # - archive/2022/heart_2022_no_nans.csv: Dataset file.
    # - static/heart_attack_model.pkl: Pretrained random forest model file.
    # - static/app.css: Custom CSS file.
    # - static/app.js: Custom JavaScript file.
    # - prompt.md: Prompt file for generating health summaries using LLMs.

    # Validate file paths:
    # Ensure the above file paths are correct within the project directory.
    # For example, heart_attack_model.pkl should be located in the static folder.
    ```

## 3. Run the Application
    ```
    # Start the application:
    # From the project root directory, run:
    python app.py

    # This starts the Uvicorn server, listening on 0.0.0.0:4242. Output will look like:
    # INFO:     Uvicorn running on http://0.0.0.0:4242 (Press CTRL+C to quit)

    # Keep the terminal window open to continue using the app.

    # Access the application:
    # Open a browser and navigate to http://localhost:4242/ to view the Gradio-based heart disease prediction and analysis interface.
    ```

## 4. Using the Application
    ```
    # Refresh data:
    # Click the "Refresh Data" button to load the latest data and generate relevant charts.

    # View data distribution:
    # Go to the "💻 Distribution" tab to see charts for state distribution, gender distribution, health status distribution, and heart attack distribution.

    # View feature importance:
    # Go to the "📊 Feature Importances" tab to view the ranking of feature importance in the model.

    # Track data:
    # Go to the "🔎 Tracking Data" tab to paginate through the dataset records.
    # Use the "◀️" and "▶️" buttons to navigate pages,
    # or enter a specific page number and click "🔄 Display" to refresh the table.

    # View risk distribution:
    # Go to the "🏃‍♂️ Risk Distribution" tab and click the "🔄 Refresh" button to update the risk distribution chart.

    # Perform risk assessment:
    # Go to the "🤖 Risk Factoring" tab to:
    # - Load data: Click "Load Random Data", "Load Healthy Data", or "Load Unhealthy Data" to auto-fill random healthy or unhealthy data.
    # - Input health factors: Manually enter health-related factors such as state, gender, health status, smoking status, etc.
    # - Calculate risk: After input, click the "Calculate Risk" button to view the risk probability and generate a health summary.
    ```

## 5. Deployment and Access
    ```
    # Public access:
    # The host parameter in app.py is set to "0.0.0.0", allowing public access.
    # - Local access: http://localhost:4242/
    # - Public access: http://<your-ip>:4242/
    ```

## 6. Additional Notes
    ```
    # Static file management:
    # Ensure the static files (CSS, JavaScript, and model files) in the static folder are correct and their paths align with app.py.

    # Data updates:
    # To update the dataset, replace the heart_2022_no_nans.csv file,
    # then click the "Refresh Data" button to reload the data.

    # Model updates:
    # To update the model, replace the heart_attack_model.pkl file (a random forest model).
    # Ensure the new model is compatible with the data preprocessing steps.
    ```
